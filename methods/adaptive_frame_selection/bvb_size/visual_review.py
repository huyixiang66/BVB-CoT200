"""Record explicit human view review; does not infer visual quality."""
import argparse
from .workflow import record_visual_review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views',required=True);p.add_argument('--out',required=True)
    p.add_argument('--decision',choices=['accept','reject'],required=True)
    p.add_argument('--notes',required=True)
    a=p.parse_args();record_visual_review(a.views,a.out,a.decision=='accept',a.notes)
    print('review recorded:',a.decision)


if __name__=='__main__':main()
