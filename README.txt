AA-HybridUNet deployment bundle (research prototype, NOT a clinical device).
models/<name>/weights.pt + model_config.json + operating_point.json ; code in aa_pneumo/.
Use in Python:  sys.path.insert(0, '<bundle>');  from aa_pneumo.infer import load_bundle, predict_image
Streamlit: unzip next to app.py so that the folder structure is  app.py, aa_pneumo/, models/<name>/...
