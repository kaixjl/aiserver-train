import yaml

target_path = "/weight/infer_cfg.yml"
cfg = yaml.load(open(target_path, "r"), Loader=yaml.SafeLoader)

if "Preprocess" in cfg:
    for it in cfg["Preprocess"]:
        if it["type"] == "NormalizeImage" and "is_scale" not in it:
            it["is_scale"] = True

yaml.dump(cfg, open(target_path, "w"))
