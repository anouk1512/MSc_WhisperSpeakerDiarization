#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jun 13 19:52:39 2023

@author: Anouk Mul 
"""

import locale
locale.getpreferredencoding = lambda: "UTF-8"
import wave, os
import pandas as pd
import time
import tracemalloc
import glob
import math
from argparse import ArgumentParser
from os import path
from huggingface_hub import notebook_login
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database import registry, get_protocol
from pyannote.audio import Model, Pipeline 
from pyannote.audio.tasks import Segmentation
from tqdm import tqdm
from types import MethodType
from torch.optim import Adam
from pytorch_lightning import Trainer
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.pipeline import Optimizer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)

notebook_login()

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir_transcription", type=str, default="data")
    parser.add_argument("--data_dir_audio", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["FRITS", "VOX"])
    parser.add_argument("--save_dir_transcriptions", type=str, default="results")
    parser.add_argument("--save_dir_performance", type=str, default="results")
    args = parser.parse_args()
    return args


def main():
    max_speakers = None
    
    # define variables 
    args = get_args()
    data_dir_transcription = args.data_dir_transcription
    data_dir_audio = args.data_dir_audio
    save_dir_transcriptions = args.save_dir_transcriptions
    save_dir_performance = args.save_dir_performance
    dataset_name = args.dataset_name
    
    # function to transform rttm annotation to correct input for pyannote 
    def get_annotation(rttm):
      anno = Annotation()
      for line in rttm:
        splitted = line.split(' ')
        total_time = float(splitted[3]) + float(splitted[4])
        anno[Segment(float(splitted[3]), total_time)] = splitted[7]
      return anno
    
    # function to get length of an audio file
    def get_length(wavpath):
      wav_obj = wave.open(wavpath, 'rb')
      sample_freq = wav_obj.getframerate()
      n_samples = wav_obj.getnframes()
      t_audio = n_samples/sample_freq
      return t_audio
    
    # function to perform diarization with pyannote pretrained pipeline 
    def pyan_diarization(audiofile, reference):
      # record start time
      start = time.time()
      tracemalloc.start()
      
      if max_speakers is not None:
        diarization = pipeline(audiofile, min_speakers=1, max_speakers=max_speakers)
      else:
        diarization = pipeline(audiofile)

      # record end time
      end = time.time()
      memory = tracemalloc.get_traced_memory() # given in (current, peak) format
      tracemalloc.stop()
    
      # get diarization error rate
      metric = DiarizationErrorRate(collar=0.25)
      dia_result = metric(reference, diarization)
    
      # print statistics
      total_time = end-start
      return memory, total_time, dia_result, diarization
    
    # function to loop through files and perform function 
    def through_files(wav_path, rttm_path): 
        results = pd.DataFrame(
          columns=["Filename", "Algorithm", "DiarizationER", "Duration",
                   "Real-time", "Size", "MemoryCurrent", "MemoryPeak"])

        #transcriptions = pd.DataFrame(columns=["Filename", "Transcription"])
        for line, subdirs, files in os.walk(wav_path):
          for name in tqdm(files):
        
            wavpath = os.path.join(line, name)
            short_name = name[:-4]
        
            # get corresponding rttm file
            rttm_name = str(short_name) + ".rttm"
            rttmpath = os.path.join(rttm_path, rttm_name)
            with open(rttmpath) as file:
              rttm_file = file.readlines()
      
            length = get_length(wavpath)
            size = os.path.getsize(wavpath)
      
            # perform speaker diarization and get statistics
            demo_mem, demo_time, demo_dia, trans = pyan_diarization(wavpath, get_annotation(rttm_file))

            results = results.append(
                {"Filename": short_name,
                 "Algorithm": "PyAnnote",
                 "DiarizationER": demo_dia,
                 "Duration": length, 
                 "Real-time": demo_time/length,
                 "Size": size, 
                 "MemoryCurrent": demo_mem[0],
                 "MemoryPeak": demo_mem[1],
                 }, ignore_index=True)

        return results
    
    
    # function to create train/dev/test database
    def create_manifest(sample_size, type_set):
      wav_path = data_dir_audio
      rttm_path = data_dir_transcription
      base_path = "/content/pyannote-audio/tests/data/debug."
    
      # specify the paths and empty the files
      rttm_output_file = base_path + type_set + ".rttm"
      open(rttm_output_file, "w").close()
    
      name_output_file = base_path + type_set + ".lst"
      open(name_output_file, "w").close()
    
      uem_output_file = base_path + type_set + ".uem"
      open(uem_output_file, "w").close()
    
      for sample in sample_size:
        sample_name = os.path.join(wav_path, sample)
        short_name = sample[:-4]
        rttm_file = short_name + ".rttm"
        rttm_name = os.path.join(rttm_path, rttm_file)
    
        # write rttm transcriptions to rttm file
        with open(rttm_name, 'r') as file:
          rttm_content = file.read()  # Read the content of the current RTTM file
          lines = rttm_content.splitlines()
          last_line = lines[-1].strip().split()
          end_time = float(last_line[3]) + math.floor(float(last_line[4]))
    
        with open(rttm_output_file, 'a') as file:
          file.write(rttm_content)  # Append the content to the output file
    
        # write file names to list file
        with open(name_output_file, "a") as file:
          file.write(short_name + "\n")
    
        # write uem to uem file
        with open(uem_output_file, "a") as file:
          file.write(f"{short_name} NA 0.000 {end_time}\n")
        
        # copy the audio file into the pyannote directory > manually done by Nils 
        #!cp "$sample_name" .
      return


    # we use Adam optimizer with 1e-4 learning rate
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)
        
        
    def get_adj_pipeline(database_path): 
        registry.load_database(database_path)
        protocol = registry.get_protocol('Debug.SpeakerDiarization.Debug')
        # model = Model.from_pretrained("pyannote/segmentation", use_auth_token=True)
        model = Pipeline.from_pretrained(f"config/pyannote_segmentation_config.yaml")

        task = Segmentation(
            protocol,
            # duration=model.specifications.duration,
            # max_num_speakers=len(model.specifications.classes),
            batch_size=32,
            num_workers=2,
            loss="bce",
            vad_loss="bce")
        model.task = task
        model.setup(stage="fit")
                
        model.configure_optimizers = MethodType(configure_optimizers, model)
        
        # we monitor diarization error rate on the validation set
        # and use to keep the best checkpoint and stop early
        monitor, direction = task.val_monitor
        checkpoint = ModelCheckpoint(
            monitor=monitor,
            mode=direction,
            save_top_k=1,
            every_n_epochs=1,
            save_last=False,
            save_weights_only=False,
            filename="{epoch}",
            verbose=False,
        )
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=direction,
            min_delta=0.0,
            patience=10,
            strict=True,
            verbose=False,
        )
        
        callbacks = [RichProgressBar(), checkpoint, early_stopping]
        
        # we train for at most 20 epochs (might be shorter in case of early stopping)
        trainer = Trainer(accelerator="gpu",
                          callbacks=callbacks,
                          max_epochs=20,
                          gradient_clip_val=0.5)
        trainer.fit(model)
                
        finetuned_model = checkpoint.best_model_path
        
        pretrained_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
        pretrained_hyperparameters = pretrained_pipeline.parameters(instantiated=True)
        
        pipeline = SpeakerDiarization(
            segmentation=finetuned_model,
            clustering="OracleClustering",
        )
        # as reported in the technical report, min_duration_off can safely be set to 0.0
        pipeline.freeze({"segmentation": {"min_duration_off": 0.0}})
        
        optimizer = Optimizer(pipeline)
        dev_set = list(protocol.development())
        
        iterations = optimizer.tune_iter(dev_set, show_progress=False)
        best_loss = 1.0
        for i, iteration in enumerate(iterations):
            print(f"Best segmentation threshold so far: {iteration['params']['segmentation']['threshold']}")
            if i > 50: break   # 50 iterations should give slightly better results
        best_segmentation_threshold = optimizer.best_params["segmentation"]["threshold"]
        
        pipeline = SpeakerDiarization(
            segmentation=finetuned_model,
            embedding=pretrained_pipeline.embedding,
            embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
            clustering=pretrained_pipeline.klustering,
        )
        
        pipeline.freeze({
            "segmentation": {
                "threshold": best_segmentation_threshold,
                "min_duration_off": 0.0,
            },
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
            },
        })
        
        optimizer = Optimizer(pipeline)
        iterations = optimizer.tune_iter(dev_set, show_progress=False)
        best_loss = 1.0
        for i, iteration in enumerate(iterations):
            print(f"Best clustering threshold so far: {iteration['params']['clustering']['threshold']}")
            if i > 50: break  # 50 iterations should give slightly better results
        best_clustering_threshold = optimizer.best_params['clustering']['threshold']
    
        finetuned_pipeline = SpeakerDiarization(
            segmentation=finetuned_model,
            embedding=pretrained_pipeline.embedding,
            embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
            clustering=pretrained_pipeline.klustering,
        )
        
        finetuned_pipeline.instantiate({
            "segmentation": {
                "threshold": best_segmentation_threshold,
                "min_duration_off": 0.0,
            },
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": best_clustering_threshold,
            },
        })
        
        return finetuned_pipeline, protocol
    
    
    def adj_pipe_metrics(pipeline, protocol): 
        metric = DiarizationErrorRate()
        
        start = time.time()
        tracemalloc.start()
        for file in protocol.test():
            # apply finetuned pipeline
            file["finetuned pipeline"] = pipeline(file)
        
            # evaluate its performance: is dit per file? 
            # hoe slaan we per file op? 
            metric(file["annotation"], file["finetuned pipeline"], uem=file["annotated"])
        end = time.time()
        memory = tracemalloc.get_traced_memory() # given in (current, peak) format
        tracemalloc.stop()
        
        adj_pipe_results = pd.DataFrame([{'Dataset': protocol.name, 'Pipeline': pipeline,'DER': metric, 'Running time': end-start, 'Memory': memory}])

        return adj_pipe_results
            
    # initiate models, need huggingface token 
    # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
    # ! Run experiments with THRESHOLDS
    for config_path in glob.glob(f"config/pyannote_diarization_config-method*.yaml"):              
      pipeline = Pipeline.from_pretrained(config_path)
      # perform pretrained pipeline on directory 
      pp_results = through_files(data_dir_audio, data_dir_transcription)
      # Save performance dataframe in folder
      # Get params
      method = "average" if "average" in config_path else "centroid"
      idx = config_path.find("seg-threshold-")+len("seg-threshold-")
      seg_thresh = config_path[idx:idx+5]
      idx = config_path.find("_cluster-threshold-")+len("_cluster-threshold-")
      cluster_thresh = config_path[idx:idx+5]
      with open(os.path.join(save_dir_performance,
                             f"pp_results_{dataset_name}_method-{method}_seg-threshold-{seg_thresh}_cluster-threshold-{cluster_thresh}.csv"),
                'w', encoding = 'utf-8-sig') as f:
          pp_results.to_csv(f, index=False)
    
    # ! Run experiments with MAX_SPEAKERS
    for max_speakers in [2, 5, 10, 15, 20]:
      pipeline = Pipeline.from_pretrained(f"config/pyannote_diarization_config-default.yaml")
      # perform pretrained pipeline on directory 
      pp_results = through_files(data_dir_audio, data_dir_transcription)
      with open(os.path.join(save_dir_performance,
                             f"pp_results_{dataset_name}_max-speakers-{max_speakers}.csv"),
                'w', encoding = 'utf-8-sig') as f:
          pp_results.to_csv(f, index=False)
    
  
  
    # ! TRAINING

    # initiate adjusted pipeline: give list of file names 
    #create_manifest(thirty_p_sample, "train")
    #create_manifest(development_set, "development")
    #create_manifest(test_set, "test")
    
    # path_to_database = "pyannote-audio-develop/tests/data/database.yml"
    # adj_pipeline, protocol = get_adj_pipeline(path_to_database)
    # ap_results = adj_pipe_metrics(adj_pipeline, protocol)
              
    # # Save transcriptions to file 
    # #trans_path = save_dir_transcriptions+("pp_transcriptions.csv")
    # #with open(trans_path, 'w', encoding = 'utf-8-sig') as f:
    # #    pp_transcriptions.to_csv(f, index = True)


    # res_path = save_dir_performance+("ap_results.csv")
    # with open(res_path, 'w', encoding = 'utf-8-sig') as f:
    #     ap_results.to_csv(f, index = True)
            

if __name__ == "__main__":
    main()