from session.cli_parser import CLIParser
from session.session import Session

if __name__ == '__main__':
    cli_args = ['-id', '123406789013', 
        '-l', 'tversky_loss', '-lp', 'beta', '0.4', 
        '-lr', '0.001', '-e', '2', '-o', 'Adam', 
        '-mc', 'UeberNet', '-mp', 'default',
        '-tg', 'default', '-vg', 'default',
        '-tts', '0.75', '-aug', 'default',
        '-svtb', 'False', '-svcp', 'False',
        '-svhis', 'False', '-svm', 'False'
       ]

    parser = CLIParser()
    args = parser.parse_args(args=cli_args)
    
    session = Session(args)
    hist = session.execute()
