import hydra

from omegaconf import (
    DictConfig, 
    OmegaConf
)

from train_epic_kitchen import main as main_epic_kitchen

@hydra.main(config_path="configs", config_name="config")
def experiment(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    main_epic_kitchen(cfg)

if __name__ == "__main__":
    experiment()