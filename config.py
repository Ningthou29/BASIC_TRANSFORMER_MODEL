from pathlib import Path



def get_config():
    return {
        "batch_size":8,
        "num_epochs": 10,
        "lr" : 10**-4,
        "d_model" : 256,
        "seq_len" : 128,
        "lang_src": "en",
        "lang_tgt": "mni",          # Change to "mni" for Meitei script, or "rom" for Romanized
        "excel_file_path": "FILE2.xlsx", # Path to your Excel file
        "model_folder" : "weights",
        "model_basename": "tmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel"
    }
def get_weights_file_path(config,epoch : str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder /model_filename)

