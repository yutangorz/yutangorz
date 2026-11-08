
# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/response_matrix')))
from pathlib import Path
import json,csv,hashlib,shutil
import numpy as np
R=_repo_path('08_COHP_ANALYSIS/workflows/response_matrix');P=_repo_path('.');S=_repo_path('STRUCTURE_PERTURBATION_EXTENSION');old=_repo_path('08_COHP_ANALYSIS/workflows/pair_windows')
def rd(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def wr(n,s):(R/n).write_text(s.strip()+'\n',encoding='utf-8')
def tab(n,rows):
 with (R/n).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((_repo_path('08_COHP_ANALYSIS/workflows/response_matrix/MATRIX_RESULTS.json')).read_text());tr=rd(_repo_path('08_COHP_ANALYSIS/workflows/pair_windows/NINE_STATE_COHP_ICOHP_TRENDS.csv'));windows=rd(_repo_path('08_COHP_ANALYSIS/workflows/pair_windows/BAND_EDGE_WINDOW_OBSERVATIONS.csv'));q=rd(_repo_path('08_COHP_ANALYSIS/workflows/pair_windows/NINE_STATE_TARGET_PAIR_QC.csv'));g=a['all_k_gram'][0]
tab('BAND_EDGE_SUBSPACE_QC.csv',[dict(k=x['k'],bands=json.dumps(x['bands']),stage=x['stage'],target_group=x['group'],trace_weight=x['trace_weight'],unitary_trace_residual=x['unitary_rotation_trace_residual'],subspace_gram_max=x['subspace_gram_max'],Gram_eigenvalues=json.dumps(x['subspace_gram_eigenvalues']),qualification='DIAGNOSTIC_TRACE_VERIFIED_NOT_PAIR_COHP_CERTIFICATION') for x in a['subspaces']]);tab('ALL_K_ORTHONORMALITY.csv',a['all_k_gram'])
tab('TARGET_ORBITAL_RELIABILITY.csv',[dict(basis_index1=i+1,label=x,source='Job13904 spin1 64k,192basis,192retained bands',validation='same row label sequence Real/Imag; metric reproduces A09 Gamma146',reliability='Gamma144 good internal orthogonality; Gamma145/146 target overlaps remain') for i,x in enumerate(a['labels']) if x.startswith(('Rh13','Cl18','Cl19'))])
wr('PROJECTION_MATRIX_ANALYSIS.md',f'''
# Job13904矩阵实际分析

实际读取三套系数及overlap矩阵，共64k×192×192复矩阵/文件。每k分实部/虚部，按`kpoint`键关联而非文件排列顺序；三个文件首k分别43、42、11，不能按块序直接相减。逐块验证192原子轨道行、实虚标签一致、有限数；能带列为1–192。源DFT为360带，192之后按冻结LOBSTER规则忽略，不能解释为360维正交性。spin1/非SOC沿已核协议。

真实标签：Rh13的4d为第65–69行，Cl18的3p为102–104行，Cl19为106–108行（1基）；不是bandOverlaps中的行号。三套文件分别为原系数及两步正交化变体，依据随包手册第678–683行。复数按Real+i Imag重建。原文件只读，SHA见MATRIX_RESULTS.json hash。

令S为原子基重叠，原/LSO1系数C使用C†SC，LSO2使用C†C。Γ S厄米残差0，最小本征值{a['S_eigenvalue_Gamma_min']:.12g}>0。LSO1的S度量与LSO2欧氏度量偏离最大值分别{g['LSO1_Snorm_max']:.12g}/{g['LSO2_Euclidean_max']:.12g}，一致支持阶段语义。Γ146行偏离{g['LSO2_row146_max']:.12g}，复现A09 0.00816559；144行{g['LSO2_row144_max']:.12g}、145行{g['LSO2_row145_max']:.12g}；占据144子块最大{g['LSO2_occupied144_max']:.12g}。全64k结果见ALL_K_ORTHONORMALITY.csv。这是实际复矩阵计算，不是abs bandOverlaps充当权重。

为在可比正交基中比较阶段，计算S^(1/2)C、S^(1/2)C_LSO1和C_LSO2。目标组权重定义为选定原子轨道行、所选能带列的模平方总和；是给定Löwdin基中的无符号子空间投影量，不等于有符号COHP或独立物理可观测量。

Γ145/146：Rh13-4d权重0.202783592→0.206670526（约+1.92%）；Cl18-3p 0.025740760→0.026019092（约+1.08%）；Cl19-3p 0.000172596→0.000598085（相对变化大但绝对量小）。这证明正交化处理确实作用于目标相关投影；不能据此直接计算COHP误差或判断哪一阶段“更真实”。LSO1与LSO2转换后权重基本一致，支持读法，并非两个独立径向基组验证。

145/146子空间内部Gram偏离约1.8568e−4，明显小于146全行0.00816559，说明部分异常耦合到子空间外，不能只因内部较好放行。对确定性QR构造的酉旋转，目标组trace残差约1e−17，正向证明该定义不依赖145/146内部选带；这一数学恒等式不证明原子基投影的绝对准确性或对外部混合稳定。

Γ144：Rh13-4d约0.305037283→0.305446545，Cl19-3p约0.049412672→0.049395737；LSO2本带范数残差约5.6e−9、全行约4.2e−8，提供单态内部正交性良好证据。但不能认证全部已占据积分，亦不能单态取代VBM近简并139–144子空间。

projectionData.lobster头包含serialization::archive，为二进制存档，手册第684–693行说明只在基组不变时保存/重载。记录头hex/大小；未用猜测结构体解码。实际矩阵分析用带显式标签的文本，避免在没有存档schema情况下猜轨道顺序。没有运行loadProjection或新LOBSTER。

PARENT缺同级系数/重叠诊断，只有原始谱、重构和64/64警告，不能由Job13904认证。L06单态/子空间正交性证据新增，但跨结构误差界、带边有符号pair COHP稳定性仍未关闭。母体与L06无足够证据证明或排除轨道身份切换；PAW PROCAR与LCAO权重定义不同，不强求相等。

执行纠错：初次用Imaginary parts匹配实际Imag parts失败，未产生数值结论；原脚本保留，v2修正后验证64k/标签和A09数值。初始把二进制当文本探测已停止，未据乱码计算。仅离线CPU分析，无VASP/LOBSTER/SLURM。
''')
im=[]
coords=json.loads((_repo_path('08_COHP_ANALYSIS/workflows/projection_checks/ANGLE_COORDINATE_REPRESENTATION_AUDIT.json')).read_text())
for x in coords:
 for rep in ['source','wrapped']:
  v=np.array(x['Cl19']);rh=np.array(x['Rh13']);v=v if rep=='source' else v-np.floor(v);img=np.round(np.array(x['Cl19'])-v).astype(int);dr=v+img-rh
  im.append(dict(state=x['state'],representation=rep,Cl19_fractional=json.dumps(v.tolist()),short_bond_image=json.dumps(img.tolist()),distance_A=float(np.linalg.norm(dr)*10.1107978),zero_image_distance_A=float(np.linalg.norm(v-rh)*10.1107978),source_SHA_verified=x['source_sha_match'],status='HOLD_PERIODIC_IMAGE' if x['state'].endswith(('ANGLE_06','ANGLE_10')) else 'GEOMETRY_MAPPING_VERIFIED'))
tab('PERIODIC_PAIR_IDENTITY_MATRIX.csv',im)
wr('ANGLE_PERIODIC_IMAGE_DIAGNOSIS.md','''
# 两种坐标表示的离线重建
PERIODIC_PAIR_IDENTITY_MATRIX.csv实际重建源坐标与[0,1)折回坐标。ANGLE06/10源零像短键2.3893688756Å；折回坐标必须(-1,0,0)才给同一短键，折回零像约10.37–10.39Å。ANGLE01/05和母体均无此x跨界。

5.1.1随包手册给出cell整数平移的语法；真实lobsterin没有cell。输出ICOHPLIST零像及短距离标签与内部折回结构的原胞零像几何未闭合。源码/内部pair相位日志缺失，不能仅由手册猜实际实现。原谱Cl19总ICOHP近零且4d–3p打印积分0；Cl18正常量级，证明异常集中于对象/通道，不证明错误由哪一内部步骤导致。真实物理响应不能凭趋势被排除；亦不能据这两数宣称断键。

实际目标轨道列表相同，未发现用户输入删除Rh4d/Cl3p；因此“主动改轨道集合”不是现有证据支持的解释。相位/周期像或正交化导致的通道效应仍需分辨。Job13904属L06，不能诊断两个角点的内部像。结论继续HOLD_PERIODIC_IMAGE，原值全保留，无标签改写。
''')
uses=['file_readability','pair_identity','occupied_ICOHP_description','Rh4d_Cl3p_qualitative_physics','VBM_mechanism','CBM_mechanism','cross_structure_quantitative_delta'];out=[]
for x in tr:
 periodic=x['state'].endswith(('ANGLE_06','ANGLE_10')) and x['pair']=='Rh13-Cl19'
 for use in uses:
  status='DESCRIPTIVE_ONLY' if use in ['file_readability','pair_identity','occupied_ICOHP_description'] else 'HOLD_PROJECTION'
  if periodic and use!='file_readability':status='HOLD_PERIODIC_IMAGE'
  out.append(dict(state=x['state'],pair=x['pair'],use=use,qualification=status,positive_evidence='SHA/columns/geometry/ICOHP and orbital-sum verified; L06 Gamma complex-matrix metric independently reproduced' if x['state']=='6313886_LENGTH_06' else 'frozen input/output identity and internal numeric reconstruction',risk='pair translation/phase not closed' if periodic else 'projection, degenerate subspace and DFT response error bounds not closed',scope='computed data description only; no validated causal mechanism' if status=='DESCRIPTIVE_ONLY' else 'no physical qualification yet'))
tab('NINE_STATE_SCIENTIFIC_QC_MATRIX.csv',out);tab('QUALIFIED_COHP_CHANNELS.csv',[x for x in out if x['qualification']=='DESCRIPTIVE_ONLY'])
rel=[]
wm={(x['state'],x['pair'],x['edge']):x for x in windows}
for x in tr:
 y=dict(x);y.update(VBM_window_COHP=wm[x['state'],x['pair'],'VBM']['COHP_window_integral'],CBM_window_COHP=wm[x['state'],x['pair'],'CBM']['COHP_window_integral'],VBM_window_d_p=wm[x['state'],x['pair'],'VBM']['Rh4d_Cl3p_window_integral'],CBM_window_d_p=wm[x['state'],x['pair'],'CBM']['Rh4d_Cl3p_window_integral'],classification='OBSERVED_DATA',energy_reference='own EF, representative sampled edge +-0.1eV; not common potential');rel.append(y)
tab('COHP_DFT_BANDGAP_RELATION.csv',rel)
def path_table(kind):
 xs=[x for x in rel if x['state']=='PARENT' or kind in x['state']]
 return '\n'.join('|'+ '|'.join([x['state'].replace('6313886_',''),x['pair'],x['local_length_A'] if kind=='LENGTH' else x['local_angle_deg'],x['ICOHP_eV_per_bond'],x['Rh4d_Cl3p_occupied_integral_eV'],str(round(float(x['delta_Eg_eV'])*1000,6))])+'|' for x in xs)
wr('LENGTH_PATH_COHP_MECHANISM.md','''
# 键长路径实际机制检验
|状态|目标键|局部长度Å|ICOHP eV/键|4d–3p占据积分eV|ΔEg meV|
|---|---|---:|---:|---:|---:|
'''+path_table('LENGTH')+'''

OBSERVED_DATA：Cl19负端→母体→正端ICOHP总体变得较不负，但正中相对母体反而更负，非单调；Cl18长度不变但ICOHP亦变化，不能将所有响应归为单键伸缩。4d–3p的Cl19占据分量负端约−0.98045→负中−0.953?→母体约−0.94744→正中−0.94343→正端−0.91546，精确值以CSV为准；总ICOHP及轨道分量口径不同，不得用一个代替另一个。

带边±0.1eV真实窗积分见关系表，原COHP正值为反键；窗口可能同时包含近邻态，Gaussian0.05eV和k积分意味着不能把曲线峰单独归属Γ145。未占据窗的pCOHP不是已占据成键能量；不能因某窗积分变大就声称它“主导”Eg。全占据积分与导带窄窗的绝对量不属于同一比较范围，不宣布价带或导带是唯一主因。

EXPLORATORY_HYPOTHESIS：Rh–Cl杂化、晶体场和非局部配位响应可能共同改变带边；A09/L06 PROCAR确认Rh-d/Cl-p参与价带，145/146近简并涉及目标Rh13/Cl18。复矩阵证明这组投影受正交化影响，尚无跨结构误差界。正负幅度非线性、Cl18共变、全局L/Theta耦合均是竞争解释。

SUPPORTED_QUALITATIVE_MECHANISM：当前仅支持在计算框架中结构与成键指标共变、且非单变量独立响应的限定描述，不构成已验的响应方向规律。UNSUPPORTED_CAUSAL_CLAIM：单键增强/削弱决定带隙、绝对VBM/CBM移动已被证实、微小ΔEg显著。own-EF只作相对谱形比较，无共同电势参考。
''')
# Remove a deliberately approximate text placeholder; use the exact frozen table value.
f=_repo_path('08_COHP_ANALYSIS/workflows/response_matrix/LENGTH_PATH_COHP_MECHANISM.md');negmid=next(x['Rh4d_Cl3p_occupied_integral_eV'] for x in tr if x['state']=='6313886_LENGTH_06' and x['pair']=='Rh13-Cl19');f.write_text(f.read_text(encoding='utf-8').replace('−0.953?',negmid),encoding='utf-8')
wr('ANGLE_PATH_COHP_MECHANISM.md','''
# 键角路径实际机制检验
|状态|目标键|局部角度°|ICOHP eV/键|4d–3p占据积分eV|ΔEg meV|
|---|---|---:|---:|---:|---:|
'''+path_table('ANGLE')+'''

OBSERVED_DATA：两键长度保持约2.389369Å，Cl18可独立保留；其4d–3p分量负端−0.94956、负中−0.94714、母体−0.94923、正中−0.94599、正端−0.94423，并非围绕母体简单对称。Cl18总ICOHP在两端/两中间也不构成按|角度|单调序列。Cl19正向两点的近零值只保留为异常，不进入正式物理趋势，也不用于相关系数。

Eg在±0.2°约均下降0.024–0.025meV，±1°约下降0.175–0.176meV；这是冻结DFT文件的近似对称观察，量级远低于尚未定界的数值误差，不能宣称验证了角度二次带隙律。没有用该对称趋势判断COHP真假。

价带/导带窗实际积分已保留。Rh-d/Cl-p杂化及晶体场改变是EXPLORATORY_HYPOTHESIS；其他Rh–Cl键及全结构Theta共变不能排除。正向Cl19像未认证，不能给双向角路径完整机制。现数据尚无独立证据将COHP变化映射为真实绝对带边移动或解释带隙因果响应；这些属UNSUPPORTED_CAUSAL_CLAIM。
''')
evidence=[('local_geometry_to_calculated_COHP','OBSERVED_DATA','real geometry and 18 pair values; 2 anomalous image channels flagged','computed branch only, no general law'),('Rh_d_Cl_p_edge_participation','OBSERVED_DATA','A09 PAW projections and actual 13904 labeled complex matrices','PAW sphere weights not signed pair COHP'),('Gamma_145_146_unitary_trace','OBSERVED_DATA','rotation residual at numerical roundoff','invariance of definition, not proof of basis accuracy'),('geometry_hybridization_bandgap','EXPLORATORY_HYPOTHESIS','co-varying geometry, orbital integrals and Eg','projection and response errors unresolved'),('single_bond_causes_Eg','UNSUPPORTED_CAUSAL_CLAIM','no isolated intervention/common reference/error bound','do not put as proven mechanism'),('framework_structural_bonding_covariation','SUPPORTED_QUALITATIVE_MECHANISM','computed observables co-vary under frozen local changes','explicit computational exploratory scope only')]
tab('MECHANISM_EVIDENCE_MATRIX.csv',[dict(claim=c,classification=k,evidence=e,boundary=b) for c,k,e,b in evidence])
wr('MINIMAL_COHP_RECALCULATION_DECISION.md','''
# 最小补算决策
默认选择A+D的限定组合：可零新增计算完成明确标注可靠性局限的计算描述，同时降低论文的已证实因果机制主张。当前没有证据证明必须补算才能发表这种限定描述。

若论文必须闭合正向角度的同一短键：条件B，首试ANGLE06一次独立LOBSTER显式周期像对照，固定原始DFT/WAVECAR、结构、192基、360带、能区、展宽和版本；手册支持`cohpBetween atom ... cell i j k orbitalWise`概念，但实际关键词组合及输入/内部坐标语义需投前核验，不能盲填(-1,0,0)或折回POSCAR套旧WAVE。一次同时追踪候选原胞/邻胞和Cl18参照；通过仅解决对象映射，不能自动解决投影；若仍不一致则降低正向Cl19主张。仅1次是首试，不保证覆盖ANGLE10，后者另审。

资源依据：原ANGLE06 48OMP约475.5s/5.323GiB；变更输出规模后耗时/内存/磁盘UNKNOWN，需实时资源和冻结包预检。该方案未生成可投递授权，执行0。已有13904不得再次作为“首次诊断输出”重跑。

条件C仅为更强带边定量机制而评估：现矩阵已证明目标子空间受投影处理影响，若主张要求比较误差，应另审原子径向基组或目标轨道稳定性测试，最小次数尚不能确定；不默认九态重算、不加NBANDS、不换SOC。矩阵的trace不变性不替代这个检验。当前无可靠误差界保证一次敏感性即关闭门，不能隐含承诺预算。

零计算替代：正文只写实际计算变化和方法局限，删去断键、单键决定Eg、绝对带边移动等强结论；原始反例完整留补充材料。所有新增DFT=0，当前新增LOBSTER=0，无提交。
''')
shutil.copyfile(_repo_path('08_COHP_ANALYSIS/workflows/response_matrix/MINIMAL_COHP_RECALCULATION_DECISION.md'),_repo_path('08_COHP_ANALYSIS/workflows/response_matrix/SINGLE_DIAGNOSTIC_RUN_DECISION.md'))
wr('FINAL_SCIENTIFIC_DECISION.md','''
# A14最终科学决定
1. 九态没有原子对获得用于“已验证跨结构成键—带边机制”的LIMITED_USE_PASS。文件可读/几何与内部数值可描述，16对保留DESCRIPTIVE_ONLY，ANGLE06/10 Cl19的物理对象继续HOLD_PERIODIC_IMAGE。没有确认FAIL；不能把无新增错误当正向认证。
2. 实际复矩阵证明正交化影响目标Rh13-4d及Cl18/Cl19-3p；Γ144内部正交性好，145/146子空间trace内旋转不变，但存在子空间外耦合和前后投影量变化。不能仅凭小内部Gram残差给机制PASS。
3. 周期像几何等价性已闭合，LOBSTER内部相互作用/相位选择尚未闭合；两者不能混为一谈。近零不是已证实的断键或已证实的软件错误。
4. ICOHP可用于明确计算及投影条件下的探索性描述。长度Cl19部分趋势与杂化假说相容，但非全程单调；Cl18亦共变。未经科学误差界认证，不用这些差值证明真实成键强弱的精确变化。
5. 数据允许描述局部几何、计算成键和Eg共变；不能完成已验证的键长/键角→带边→Eg因果证据链。带边0.444meV近简并不能用0.05eV展宽谱的单峰替代。
6. Rh–Cl杂化、晶体场、非局部配位和带边切换均保留假说；未独立排除其他机制。自身EF对齐不证明绝对带边移动。
7. 对降低后的描述性主张最低新增0；若必须闭合正角Cl19对象，条件首试1次ANGLE06显式像LOBSTER，另态/投影敏感性另审。DFT0，不自动执行。
8. 可以在正文如实注明上述限制、不给已验证机制标签的情况下结束本轮COHP分析。若作者坚持更强机制主张，则本轮科学门仍HOLD，不能宣称研究机制已经证实。

本轮真正处理了13904三套系数和S矩阵、64k/192轨道复数据，输出带边子空间与轨道索引、两种周期表示、九态18对×7用途126项分级及原36窗口联表。projectionData二进制仅认证格式来源，不臆测schema；母体无同级系数，未自动转移L06认证。未生成图件，未改旧QC/原始结果，新增DFT/LOBSTER/SLURM0。完整离线可验证证据已归档；内部映射与物理准确性问题明确保留，停止等待作者决定。
''')
wr('CHECKPOINT.md','''
# run108 A14
已完成现有13904三套系数/S矩阵实际分析：64k、spin1、192轨道/保留带，复数标签与k顺序核验，Γ146行0.0081655899复现A09；144良好但145/146目标Rh/Cl子空间投影受正交化影响。子空间trace旋转不变是正向数学证据，不认证跨结构物理准确性。
九态18对×7用途126项；LIMITED_USE_PASS0、确认FAIL0；16对数据描述保留，两正角Cl19像仍HOLD。几何周期等价已证，内部pair/相位未证。机制数值及36窗已联表，正式因果机制不成立；零新增可完成限定描述，条件1次ANGLE06像对照另审。新DFT/LOBSTER/提交/绘图修改0。旧A13/A09/A11原样保留。详见FINAL_SCIENTIFIC_DECISION.md，项目未完成，停止待作者。
''')
inputs=[_repo_path('08_COHP_ANALYSIS/workflows/pair_windows/NINE_STATE_COHP_ICOHP_TRENDS.csv'),_repo_path('08_COHP_ANALYSIS/workflows/pair_windows/BAND_EDGE_WINDOW_OBSERVATIONS.csv'),_repo_path('08_COHP_ANALYSIS/workflows/pair_windows/REMOTE_ANALYSIS.json'),_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/LOBSTER_MANUAL_EXTRACT.txt'),_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/FINAL_ACCEPTANCE_DECISION.md'),_repo_path('08_COHP_ANALYSIS/workflows/projection_checks/ANGLE_COORDINATE_REPRESENTATION_AUDIT.json')]
tab('SOURCE_SHA256.csv',[dict(path=str(p),sha256=sha(p)) for p in inputs]);tab('MATRIX_SOURCE_SHA256.csv',[dict(file=n,sha256=h,remote_directory=str(_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_a09_length06_diag_run103_20261030/OUTPUT'))) for n,h in a['hash'].items()])
wr('VALIDATION.json',json.dumps(dict(kpoints=64,basis_functions=192,retained_bands=192,complex_matrices=256,states=9,pair_use_decisions=len(out),limited_physical_PASS=0,confirmed_FAIL=0,maximum_rotation_trace_residual=max(x['unitary_rotation_trace_residual'] for x in a['subspaces']),new_DFT=0,new_LOBSTER=0,plots_modified=0,project_complete=False),indent=2))
print('A14 outputs',len(out))
