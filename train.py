from mwm.components.training import Training


def main():
    trainer = Training()
    trainer.make_model()
    trainer.handle_device()
    trainer.make_dataset()
    trainer.make_criterion()
    trainer.make_optimizer()
    trainer.train(save_model=True, save_interval=1)

if __name__ == "__main__":
    main()