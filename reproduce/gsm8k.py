"""Pinned public GSM8K data, separate calibration, and explicit numeric scoring."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import workload

REVISION='3101c7d5072418e28b9008a6636bde82a006892c'
BASE='https://raw.githubusercontent.com/openai/grade-school-math/'+REVISION+'/'
FILES={'train.jsonl':'grade_school_math/data/train.jsonl',
       'test.jsonl':'grade_school_math/data/test.jsonl','LICENSE':'LICENSE'}
SCORER='gsm8k-final-number-v1'
NUMBER=r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'


def source_lock():
    lock=json.loads((Path(__file__).resolve().parents[1]/'configs/gsm8k-source.json').read_text())
    if lock.get('revision')!=REVISION or lock.get('scorer')!=SCORER:
        raise ValueError('GSM8K source lock differs from implementation')
    return lock


def final_number(text):
    if not isinstance(text,str): return None
    markers=list(re.finditer(r'(?m)^\s*####\s*('+NUMBER+r')\s*$',text))
    if len(markers)!=1 or text[markers[0].end():].strip(): return None
    try:
        value=Decimal(markers[0].group(1).replace(',',''))
        return value if value.is_finite() else None
    except InvalidOperation:return None


def records(data,expected_count):
    rows=[json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    if len(rows)!=expected_count or any(set(x)!={'question','answer'} or
        not isinstance(x['question'],str) or not x['question'].strip() or
        final_number(x['answer']) is None for x in rows):
        raise ValueError('GSM8K source row count or answer schema mismatch')
    return rows


def calibration(train,test,count=50):
    excluded={x['question'] for x in test};seen=set();chosen=[]
    order=sorted(range(len(train)),key=lambda i:hashlib.sha256(
        ('dsv41-gsm8k-calibration-v1:'+train[i]['question']).encode()).hexdigest())
    for index in order:
        row=train[index]
        if row['question'] in excluded or row['question'] in seen:continue
        seen.add(row['question']);chosen.append({'source_split':'train','source_index':index,**row})
        if len(chosen)==count:return chosen
    raise ValueError('insufficient calibration questions distinct from scored test')


def fetch(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    payloads={}
    for name,path in FILES.items():
        with urllib.request.urlopen(BASE+path,timeout=30) as response:data=response.read(12_000_001)
        if len(data)>12_000_000:raise ValueError('GSM8K source exceeds size bound')
        payloads[name]=data
    if b'MIT License' not in payloads['LICENSE'] or b'Copyright (c) 2021 OpenAI' not in payloads['LICENSE']:
        raise ValueError('source license could not be verified')
    train=records(payloads['train.jsonl'],7473);test=records(payloads['test.jsonl'],1319)
    payloads['calibration.json']=(json.dumps(calibration(train,test),ensure_ascii=True,sort_keys=True,indent=2)+'\n').encode()
    if {name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}!=source_lock()['files']:
        raise ValueError('downloaded GSM8K data differs from pinned source hashes')
    for name,data in payloads.items():
        with (root/name).open('xb') as output:output.write(data)
    manifest={'source':'openai/grade-school-math','revision':REVISION,'license':'MIT',
              'sources':{name:BASE+path for name,path in FILES.items()},
              'files':{name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()},
              'scored_split':'test','scored_count':1319,'calibration_split':'train',
              'calibration_count':50,'scorer':SCORER,
              'scope':'public source data and answer oracle; no model results'}
    (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return verify(root)


def verify(destination):
    root=Path(destination);m=json.loads((root/'manifest.json').read_text())
    expected=set(FILES)|{'calibration.json'}
    if (m.get('source')!='openai/grade-school-math' or m.get('revision')!=REVISION or m.get('scorer')!=SCORER or
        m.get('license')!='MIT' or m.get('sources')!={name:BASE+path for name,path in FILES.items()} or
        m.get('scored_count')!=1319 or m.get('calibration_count')!=50 or
        m.get('scored_split')!='test' or m.get('calibration_split')!='train' or m.get('files')!=source_lock()['files'] or set(m.get('files',{}))!=expected or
        {x.name for x in root.iterdir()}!=expected|{'manifest.json'}):
        raise ValueError('GSM8K dataset provenance or file coverage mismatch')
    for name,digest in m['files'].items():
        f=root/name
        if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
            raise ValueError('GSM8K dataset hash mismatch')
    train=records((root/'train.jsonl').read_bytes(),7473);test=records((root/'test.jsonl').read_bytes(),1319)
    if json.loads((root/'calibration.json').read_text())!=calibration(train,test):
        raise ValueError('GSM8K calibration differs from registered selection')
    return {'revision':REVISION,'scored_count':len(test),'calibration_count':50,
            'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}


def score(answer,response,finish_reason='stop',complete=True,http_status=200):
    gold=final_number(answer)
    if gold is None:raise ValueError('invalid source answer oracle')
    failure=not complete or http_status!=200
    truncated=finish_reason=='length'
    parsed=final_number(response) if not failure and finish_reason=='stop' else None
    return {'scorer':SCORER,'correct':bool(parsed is not None and parsed==gold),
            'extraction_failed':bool(not failure and not truncated and parsed is None),
            'truncated':truncated,'request_failed':failure,
            'denominator':1}


def task(tokenizer,row,index,thinking,cap,split='test'):
    if type(index) is not int or index<0 or type(thinking) is not bool or split not in ('test','calibration'):
        raise ValueError('invalid GSM8K task identity or condition')
    if final_number(row['answer']) is None:raise ValueError('invalid GSM8K source oracle')
    messages=[{'role':'system','content':'Solve the math problem. End your response with a single line: #### followed by the numeric answer. Do not write anything after that line.'},
              {'role':'user','content':row['question']}]
    rendered=tokenizer.apply_chat_template(messages,thinking=thinking,reasoning_effort=50,tokenize=False)
    tokens=len(tokenizer.encode(rendered,add_special_tokens=False))
    seed=workload.seed_for('gsm8k-'+split+'-'+('on' if thinking else 'off'),0,8,0,index)
    return {'id':('gt' if split=='test' else 'gc')+str(index),'suite':'gsm8k','split':split,
            'source_index':index,'dataset_revision':REVISION,'scorer':SCORER,
            'prompt_version':'gsm8k-zero-shot-final-number-v1','thinking':thinking,'effort':50,
            'body':workload.request_body('/model',messages,tokens,cap,seed,thinking=thinking,effort=50,fixed_output=False),
            'input_tokens':tokens,'output_tokens':cap,'expected_answer':row['answer'],
            'prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['fetch','verify','prepare']);parser.add_argument('directory')
    parser.add_argument('--model');parser.add_argument('--thinking',choices=['on','off']);parser.add_argument('--cap',type=int)
    parser.add_argument('--split',choices=['test','calibration'],default='test')
    args=parser.parse_args()
    if args.operation=='prepare':
        if not args.model or args.thinking is None or not args.cap:parser.error('prepare requires model, thinking and cap')
        verify(args.directory)
        from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
        tokenizer=DeepseekV41Tokenizer.from_pretrained(args.model,local_files_only=True)
        root=Path(args.directory)
        rows=records((root/'test.jsonl').read_bytes(),1319) if args.split=='test' else json.loads((root/'calibration.json').read_text())
        result=[task(tokenizer,row,index if args.split=='test' else row['source_index'],args.thinking=='on',args.cap,args.split)
                for index,row in enumerate(rows)]
    else:result=fetch(args.directory) if args.operation=='fetch' else verify(args.directory)
    print(json.dumps(result,sort_keys=True))
