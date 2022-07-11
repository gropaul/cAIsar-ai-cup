from argparse import Action, ArgumentParser, Namespace

class StoreSingleValueAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace, values, option_string = None) -> None:
        setattr(namespace, self.dest, values[0])