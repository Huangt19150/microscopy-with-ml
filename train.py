from mwm.components.training import Training


def main():
    trainer = Training()
    trainer.handle_device()
    trainer.make_criterion()
    trainer.make_optimizer()
    trainer.train(save_model=True, save_interval=1)

if __name__ == "__main__":
    main()