

def check_main_args(args):

    if not args.train:

        if args.title is None or args.title == '':
            raise AttributeError('No valid argument provided as description')

        if args.description is None or args.description == '':
            raise AttributeError('No valid argument provided as description')