"""Export Student/EMA with actual algorithm, source and runtime configuration."""
import argparse,json
from pathlib import Path
from src.dimo.infer import load_role
from src.dimo.origin import sha256,weight_files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model',choices=['student','ema'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Export refuses to overwrite an existing directory')
    model,metadata=load_role(args.checkpoint,args.model)
    model.save_pretrained(args.output,safe_serialization=True)
    metadata['checksums']={f.name:sha256(f) for f in [args.output/'config.json',*weight_files(args.output)]}
    metadata['inference_defaults']=dict(cfg=1,steps=1,noise_emb_perturb=0.0,output_compositing='none')
    (args.output/'export.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
