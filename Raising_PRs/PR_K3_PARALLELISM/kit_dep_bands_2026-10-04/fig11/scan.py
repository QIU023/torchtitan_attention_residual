from PIL import Image
im = Image.open('p19-19.png').convert('RGB')
k = 500/72
t0, u = 98.05, 5.427
NAMES = {
 (238,243,251):'F0',(175,210,242):'F1',(95,168,228):'F2',(47,111,208):'F3',
 (131,33,25):'B3',(189,78,48):'B2',(225,126,82):'B1',(233,154,107):'B0',
 (213,232,212):'VF',(96,169,23):'VB',(255,255,255):'.',(249,247,237):'DL',
}
def name(c):
    best=min(NAMES,key=lambda n: sum((a-b)**2 for a,b in zip(n,c)))
    d=sum((a-b)**2 for a,b in zip(best,c))
    return NAMES[best] if d<400 else str(c)
rows={}
for rname, yscan in (('PP0',116.5),('PP1',132.8),('PP2',149.0)):
    y = int(yscan*k)
    segs=[]; cur=None
    for x in range(int(97*k), int(545*k)):
        c = im.getpixel((x,y))
        key = 'border' if sum(c) < 60 else name(c)
        if cur is None or key!=cur[0]:
            if cur: segs.append(cur)
            cur=[key,x,x]
        else: cur[2]=x
    segs.append(cur)
    out=[]
    for key,a,b in segs:
        if key=='border': continue
        s=((a/k)-t0)/u; e=(((b+1)/k)-t0)/u
        if e-s<0.2: continue
        out.append((round(s*2)/2, round(e*2)/2, key))
    rows[rname]=out
if __name__=='__main__':
    for r,out in rows.items():
        print(r, ' '.join(f'{k}[{s:g},{e:g})' for s,e,k in out))
