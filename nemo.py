# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:09:00 2023

@author: Anouk Mul 
"""

from argparse import ArgumentParser
import tracemalloc
import time 
import pandas as pd 
import nemo.collections.asr as nemo_asr
import numpy as np
import librosa
import os
import wget
import matplotlib.pyplot as plt
import nemo
import glob
import json
import shutil

from IPython.display import Audio, display
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.collections.asr.models.msdd_models import NeuralDiarizer


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir_transcription", type=str, default="data")
    parser.add_argument("--data_dir_audio", type=str, default="data")
    parser.add_argument("--save_dir_performance", type=str, default="results")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["FRITS", "VOX"])
    parser.add_argument("--domain_type", type=str, default="telephonic")
    parser.add_argument("--sigmoid_threshold", type=float, default=0.7)
    # Added
    parser.add_argument("--manifest_path", type=str, default="config/manifest.json")
    parser.add_argument("--base_dir", type=str, default="temporary_files")
    args = parser.parse_args()
    return args


def main():
    # define variables 
    max_speakers = 8
    args = get_args()
    data_dir_transcription = args.data_dir_transcription
    data_dir_audio = args.data_dir_audio
    dataset_name = args.dataset_name
    save_dir_performance = args.save_dir_performance
    os.makedirs(save_dir_performance, exist_ok=True)
    domain_type = args.domain_type
    sigmoid_threshold = args.sigmoid_threshold
    
    DOMAIN_TYPE = domain_type
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    BASE_DIR = args.base_dir
    MANIFEST_PATH = args.manifest_path
    
    CONFIG_URL = f"NeMo-main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"

    if not os.path.exists(os.path.join(BASE_DIR,CONFIG_FILE_NAME)):
        CONFIG = wget.download(CONFIG_URL, BASE_DIR)
    else:
        CONFIG = os.path.join(BASE_DIR,CONFIG_FILE_NAME)
        
    cfg = OmegaConf.load(CONFIG)
    cfg.diarizer.manifest_filepath = MANIFEST_PATH
      
    def nemo_diarization_clust(MANIFEST_PATH): 
      start = time.time()
      tracemalloc.start()

      pretrained_speaker_model='titanet_large'
    
      # Directory to store intermediate files and prediction outputs
      cfg.diarizer.out_dir = BASE_DIR 
      cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
      cfg.diarizer.clustering.parameters.oracle_num_speakers=False
      cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
    
    
      # Using neural VAD 
      cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet' # deze ook nodig
      cfg.diarizer.clustering.max_num_speakers = max_speakers

      
      # USE CLUSTERING DIARIZER - STAP 1 
      sd_model = ClusteringDiarizer(cfg=cfg)
      results = sd_model.diarize()
      print(results[-1])
      
      # record end time
      end = time.time()
      memory = tracemalloc.get_traced_memory() # given in (current, peak) format 
      tracemalloc.stop()
    
      # !zip -r /content/pred_rttms_clustering.zip /content/pred_rttms
      # files.download("/content/pred_rttms_clustering.zip")
      # files.download("/content/diar_infer_meeting.yaml")
    
      # store the data and export it 
      df = pd.DataFrame([{
        "DiarizationER": results[-1][0],
        "ConfusionER": results[-1][1],
        "FalseAlarm": results[-1][2],
        "MissedSpeech": results[-1][3],
        "Duration": end-start,
        "MemoryCurrent": memory[0],
        "MemoryPeak": memory[1]}])
      # df = pd.DataFrame([{'Scores': results[-1], 'Duration': end-start, 'Memory': memory}])
    
      del sd_model 
      return df 
  
    def nemo_diarization_neural(MANIFEST_PATH): 
      start = time.time()
      tracemalloc.start()

      pretrained_speaker_model='titanet_large'
      #cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
    
      # Directory to store intermediate files and prediction outputs
      cfg.diarizer.out_dir = BASE_DIR 
      cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
      cfg.diarizer.clustering.parameters.oracle_num_speakers=False
      cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
    
      # print de output van bovenstaand
      #!cat {cfg.diarizer.manifest_filepath}
    
      # Using neural VAD 
      cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet' # deze ook nodig
      
      # USE CLUSTERING DIARIZER - STAP 1 
      sd_model = ClusteringDiarizer(cfg=cfg)
      sd_model.diarize()
      
      # USE NEURAL DIARIZER - DIT AANZETTEN IS STAP 2 
      cfg.diarizer.msdd_model.model_path = 'diar_msdd_telephonic' # Telephonic speaker diarization model 
      cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [sigmoid_threshold]
      system_vad_msdd_model = NeuralDiarizer(cfg=cfg)
      
      # How to store result? How to store specific settings? 
      results = system_vad_msdd_model.diarize()[0]
    
      # record end time
      end = time.time()
      memory = tracemalloc.get_traced_memory() # given in (current, peak) format 
      tracemalloc.stop()
    
      # !zip -r /content/pred_rttms_clustering.zip /content/pred_rttms
      # files.download("/content/pred_rttms_clustering.zip")
      # files.download("/content/diar_infer_meeting.yaml")
    
      # store the data and export it 
      # df = pd.DataFrame([{'Scores': results, 'Duration': end-start, 'Memory': memory}])

      df = pd.DataFrame([{
        "DiarizationER": results[-1][0],
        "ConfusionER": results[-1][1],
        "FalseAlarm": results[-1][2],
        "MissedSpeech": results[-1][3],
        "Duration": end-start,
        "MemoryCurrent": memory[0],
        "MemoryPeak": memory[1]}])
      
      del sd_model 
      del system_vad_msdd_model
      return df
    
    
    def nemo_diarization_variation(MANIFEST_PATH): 
      start = time.time()
      tracemalloc.start()

      pretrained_speaker_model='titanet_large'
      #cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
    
      # Directory to store intermediate files and prediction outputs
      cfg.diarizer.out_dir = BASE_DIR 
      cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
      cfg.diarizer.clustering.parameters.oracle_num_speakers=False
      cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
    
      # print de output van bovenstaand
      #!cat {cfg.diarizer.manifest_filepath}
    
      # Using neural VAD 
      cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet' # deze ook nodig
      
      # hiermee varieren 
      cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
      cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
      cfg.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1]
      
      # USE CLUSTERING DIARIZER - STAP 1 
      sd_model = ClusteringDiarizer(cfg=cfg)
      results = sd_model.diarize()
    
      # record end time
      end = time.time()
      memory = tracemalloc.get_traced_memory() # given in (current, peak) format 
      tracemalloc.stop()
    
      # !zip -r /content/pred_rttms_clustering.zip /content/pred_rttms
      # files.download("/content/pred_rttms_clustering.zip")
      # files.download("/content/diar_infer_meeting.yaml")
    
      # store the data and export it 
      # df = pd.DataFrame([{'Scores': results, 'Duration': end-start, 'Memory': memory}]) 
      df = pd.DataFrame([{
        "DiarizationER": results[-1][0],
        "ConfusionER": results[-1][1],
        "FalseAlarm": results[-1][2],
        "MissedSpeech": results[-1][3],
        "Duration": end-start,
        "MemoryCurrent": memory[0],
        "MemoryPeak": memory[1]}])
      
      del sd_model 
      
      return df
  
    
    def nemo_asr_diarization(MANIFEST_PATH):
      start = time.time()
      tracemalloc.start()

      pretrained_speaker_model='titanet_large'
      cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
      cfg.diarizer.out_dir = BASE_DIR #Directory to store intermediate files and prediction outputs
      cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
      cfg.diarizer.clustering.parameters.oracle_num_speakers=False
    
      # Using Neural VAD and Conformer ASR
      cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
      cfg.diarizer.asr.model_path = 'stt_en_conformer_ctc_large'
      cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD
      cfg.diarizer.asr.parameters.asr_based_vad = False
    
      asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
      asr_model = asr_decoder_ts.set_asr_model()
      word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
    
      asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
      asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
    
      diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
    
      # record end time
      end = time.time()
      memory = tracemalloc.get_traced_memory() # given in (current, peak) format
      tracemalloc.stop()
      
      # store the data and export it 
      # df = pd.DataFrame([{'Scores': diar_score, 'Duration': end-start, 'Memory': memory}])
      df = pd.DataFrame([{
        "DiarizationER": results[-1][0],
        "ConfusionER": results[-1][1],
        "FalseAlarm": results[-1][2],
        "MissedSpeech": results[-1][3],
        "Duration": end-start,
        "MemoryCurrent": memory[0],
        "MemoryPeak": memory[1]}])
      
      del asr_model

      return df
  
    # ! EXPERIMENTS
    for domain_type in ["telephonic", "meeting"]:
        results = nemo_diarization_clust(MANIFEST_PATH)
        with open(os.path.join(save_dir_performance,
                               f"nemo_results_nemo_diarization_clust_{dataset_name}_domain-type-{domain_type}.csv"),
                'w', encoding = 'utf-8-sig') as f:
            results.to_csv(f, index=False)

    for sigmoid_threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        results = nemo_diarization_neural(MANIFEST_PATH)
        with open(os.path.join(save_dir_performance,
                               f"nemo_results_nemo_diarization_neural_{dataset_name}_sigmoid-threshold-{sigmoid_threshold}.csv"),
                'w', encoding = 'utf-8-sig') as f:
            results.to_csv(f, index=False)
            
    for max_speakers in [2, 5, 10, 15, 20]:
        results = nemo_diarization_clust(MANIFEST_PATH)
        with open(os.path.join(save_dir_performance,
                               f"nemo_results_nemo_diarization_clust_{dataset_name}_max-speakers-{max_speakers}.csv"),
                'w', encoding = 'utf-8-sig') as f:
            results.to_csv(f, index=False)

    results = nemo_asr_diarization(MANIFEST_PATH)
    with open(os.path.join(save_dir_performance,
                           f"nemo_results_nemo_asr_diarization_{dataset_name}.csv"),
            'w', encoding = 'utf-8-sig') as f:
        results.to_csv(f, index=False)

  
  

    # ! DEFAULT
    # for name, func_name in [("clust_results", nemo_diarization_clust),
    #                         ("neur_results", nemo_diarization_neural),
    #                         ("adj_clust_results", nemo_diarization_variation),
    #                         ("asr_results", nemo_asr_diarization),
    #                         ]:
    #     results = func_name(MANIFEST_PATH)
    #     with open(os.path.join(save_dir_performance, f"{name}.csv"),
    #             'w', encoding = 'utf-8-sig') as f:
    #         results.to_csv(f, index=False)

    # clust_results = nemo_diarization_clust(MANIFEST_PATH)
    # # Save transcriptions to file 
    # trans_path = save_dir_performance+()
    # with open(os.path.join(save_dir_performance, "clust_results.csv"),
    #           'w', encoding = 'utf-8-sig') as f:
    #     clust_results.to_csv(f, index = True)

    # neur_results = nemo_diarization_neural(MANIFEST_PATH)
    # # Save transcriptions to file 
    # with open(os.path.join(save_dir_performance, "neur_results.csv"),
    #           'w', encoding = 'utf-8-sig') as f:
    #     neur_results.to_csv(f, index = True)

    # adj_clust_results = nemo_diarization_variation(MANIFEST_PATH)
    # # Save transcriptions to file 
    # with open(os.path.join(save_dir_performance, "adj_clust_results.csv"),
    #           'w', encoding = 'utf-8-sig') as f:
    #     adj_clust_results.to_csv(f, index = True)
        
    # asr_results = nemo_asr_diarization(MANIFEST_PATH)
    # # Save transcriptions to file 
    # with open(os.path.join(save_dir_performance, "asr_results.csv"),
    #           'w', encoding = 'utf-8-sig') as f:
    #     asr_results.to_csv(f, index = True)


if __name__ == "__main__":
    main()