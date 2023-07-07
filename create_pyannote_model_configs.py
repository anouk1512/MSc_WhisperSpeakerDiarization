template = """
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    segmentation: /home/nilsh/gitlab_haas_projects/whisper-speech-to-text/anouk/config/models/pytorch_segmentation_model.bin
    segmentation_step: 0.1
    segmentation_batch_size: 32
    embedding: /home/nilsh/gitlab_haas_projects/whisper-speech-to-text/anouk/config/models/pytorch_embedding_model.bin
    embedding_exclude_overlap: True
    embedding_batch_size: 32
    clustering: AgglomerativeClustering

params:
  segmentation:
    min_duration_off: 0.5817029604921046
    threshold: segmentation_threshold
  clustering:
    method: clustering_method
    threshold: clustering_threshold
    min_cluster_size: 15
"""

for method in ["average", "centroid"]:
    for t_seg in range(30, 65, 5):
        for t_cluster in range(50, 80, 5):
            with open(f"config/pyannote_diarization_config-method-{method}_seg-threshold-{t_seg/100:.3f}_cluster-threshold-{t_cluster/100:.3f}.yaml", "w") as f:
                f.write(template.replace("segmentation_threshold", str(t_seg/100)
                                         ).replace("clustering_threshold", str(t_cluster/100)
                                                   ).replace("clustering_method", method))