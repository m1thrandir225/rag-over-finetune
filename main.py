from internal.config import ConfigLoader

config_path = "./config.json"


def main():
    config_loader = ConfigLoader(config_path)

    config = config_loader.load_config()
    print(config)


if __name__ == "__main__":
    main()
