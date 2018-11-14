# External Modules
from argparse   import ArgumentParser
from os         import listdir
from os.path    import isdir,join
# Internal modules
from graphatoms.chargemol.submitter_untyped import BondAnalyzer
########################################################################
def main(args:dict)->None:

    ba = BondAnalyzer(dftcode=args['dftcode'],quality=args['quality'])
    root = args['pth']
    traj = args['name']+'.traj'
    assert isdir(root)
    children = listdir(root)
    if traj in children:
        pths = [join(root,traj)]
    else:
        pths = []
        for c in children:
            child = join(root,c)
            if isdir(child) and traj in listdir(join(root,c)):
                pths.append(child)

    for pth in pths:
        if args['submit']:
            ba.submit(working_path=pth,trajname=args['name'])
        else:
            ba.analyze(working_path=pth,trajname=args['name'])

if __name__ == '__main__':
    parser = ArgumentParser(description  = 'Submit chargemol jobs',
                            allow_abbrev = True)

    parser.add_argument('--pth',
                        default = str,
                        help    = 'A directory with a .traj or one level above')

    parser.add_argument('--submit',
                        default = True,
                        help    = 'Submit as batch job')

    parser.add_argument('--name',
                        default = str,
                        help    = 'X.traj')

    parser.add_argument('--quality',
                        default = 'low',
                        help    = 'Quality of DFT calculation')

    parser.add_argument('--dftcode',
                        default = 'gpaw',
                        help    = 'How to generate charge density')

    args = parser.parse_args()
    main(vars(args))
