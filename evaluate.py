from mwm.components.evaluation import Evaluator

def main():
    evaluator = Evaluator()
    evaluator.handle_device()
    evaluator.evaluate()

if __name__ == "__main__":
    main()