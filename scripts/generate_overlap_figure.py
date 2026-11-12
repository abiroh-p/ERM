from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
REPO=Path(__file__).resolve().parents[1]
R=REPO/'results/paired'
fig,ax=plt.subplots(figsize=(8.8,3.1));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
boxes=[(.02,.62,.21,.29,'#dceaf5','ISOT\n38,030 cleaned'),(.02,.12,.21,.29,'#dceaf5','McIntire\n6,005 cleaned'),(.42,.37,.21,.31,'#f5e5cb','WELFake\n61,734 cleaned'),(.76,.37,.22,.31,'#d6eed8','WELFake residual\n17,162 documents\n10,427 real / 6,735 fake')]
for x,y,w,h,color,label in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009,rounding_size=.018',facecolor=color,edgecolor='#444',lw=1))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.7)
for a,b in [((.235,.755),(.41,.61)),((.235,.275),(.41,.43)),((.64,.525),(.75,.525))]:
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#444',lw=1.1))
ax.text(.33,.78,'38,030 of 38,030\nfound in WELFake',ha='center',fontsize=9)
ax.text(.33,.10,'6,005 of 6,005\nfound in WELFake',ha='center',fontsize=9)
ax.text(.125,.50,'ISOT / McIntire:\n2 exact and about 33\nnear-duplicate matches',ha='center',va='center',fontsize=8.6,color='#555')
ax.text(.695,.71,'44,572 flagged\nrows removed',ha='center',fontsize=9)
ax.text(.87,.15,'One retained exact-match pair;\nneither article selected in\nthe paired-validation runs',ha='center',va='center',fontsize=9,color='#444')
fig.tight_layout(pad=.5)
for ext in ['pdf','png','svg']:
 fig.savefig(R/'figures'/('audit_reconciled_overlap.'+ext),dpi=250,bbox_inches='tight')
plt.close(fig)

import shutil
shutil.copy2(R/'figures/audit_reconciled_overlap.pdf',REPO/'paper/figs/fig_overlap.pdf')
