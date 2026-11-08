
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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/pair_windows')))
from pathlib import Path
import json,csv,hashlib,shutil,re
import numpy as np
R=_repo_path('08_COHP_ANALYSIS/workflows/pair_windows');P=_repo_path('.');A=_repo_path('08_COHP_ANALYSIS/workflows/original_spectra');S=_repo_path('STRUCTURE_PERTURBATION_EXTENSION');a=json.loads((_repo_path('08_COHP_ANALYSIS/workflows/pair_windows/REMOTE_ANALYSIS.json')).read_text());remote={x['state']:x for x in a['results']}
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(n,s):(R/n).write_text(s.strip()+'\n',encoding='utf-8')
def table(n,r):
 with (R/n).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta={x['state']:x for x in read(_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/NINE_STATE_STRUCTURE_AUDIT.csv'))};edge={(x['state'],x['edge']):x for x in read(_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/BAND_EDGE_RECOVERY.csv'))};pairinfo={}
for x in read(_repo_path('08_COHP_ANALYSIS/workflows/projection_checks/TARGET_PAIR_ICOHP_AUDIT.csv')):pairinfo[(x['state'],int(x['Cl_1based']))]=x
for x in read(_repo_path('08_COHP_ANALYSIS/workflows/original_spectra/ORIGIN_2024_DATA/ICOHP_COMPARISON.csv')):pairinfo[(x['state'] if x['state']=='PARENT' else '6313886_LENGTH_06',int(x['pair'].split('Cl')[1]))]=dict(ICOHP_eV_per_bond=x['ICOHP_eV_per_bond'],source_distance_A=x['distance_A'])
trend=[];qc=[];risks=[];windows=[];sources=[]
for state,m in meta.items():
 for cl in [19,18]:
  v=pairinfo[(state,cl)];periodic=state.endswith(('ANGLE_06','ANGLE_10')) and cl==19
  if state in remote:
   rp=next(x for x in remote[state]['pairs'] if 'Cl'+str(cl)+'(' in x['label']);dp=rp['Rh4d_Cl3p_ICOHP'];samples=rp['window_samples'];E=np.array([x['E'] for x in samples]);total=np.array([x['COHP'] for x in samples]);dpy=np.array([x['Rh4d_Cl3p'] for x in samples])
  else:
   name='PARENT' if state=='PARENT' else 'LENGTH_06';f=_repo_path('08_COHP_ANALYSIS/workflows/original_spectra/ORIGIN_2024_DATA')/f'{name}_Rh13_Cl{cl}_ORBITAL_GROUPS.csv';z=np.loadtxt(f,delimiter=',',skiprows=1,encoding='utf-8-sig');E=z[:,0];dpy=-z[:,-1];dp=float(np.trapz(dpy[E<=0],E[E<=0]));comp=np.loadtxt(_repo_path('08_COHP_ANALYSIS/workflows/original_spectra/ORIGIN_2024_DATA/PARENT_LENGTH06_COMPARISON.csv'),delimiter=',',skiprows=1,encoding='utf-8-sig');ix=1+(0 if name=='PARENT' else 8)+(0 if cl==19 else 4);total=comp[:,ix]
  for ed in ['VBM','CBM']:
   b=edge[(state,ed)];center=float(b['relative_to_own_EF_eV']);mask=(E>=center-.1)&(E<=center+.1)
   windows.append(dict(state=state,pair=f'Rh13-Cl{cl}',edge=ed,window_half_width_eV=.1,energy_reference='each own EF; representative sampled edge, no common potential',center_eV=center,COHP_window_integral=float(np.trapz(total[mask],E[mask])),Rh4d_Cl3p_window_integral=float(np.trapz(dpy[mask],E[mask])),sample_count=int(mask.sum()),interpretation='FILE_NUMERICAL_DESCRIPTION_NOT_K_RESOLVED_PAIR_ATTRIBUTION',qc='HOLD_PERIODIC_IMAGE' if periodic else 'HOLD_UNRESOLVED_PROJECTION'))
  base=float(pairinfo[('PARENT',cl)]['ICOHP_eV_per_bond']);cur=float(v['ICOHP_eV_per_bond'])
  trend.append(dict(state=state,source_DFT_job=m['job_id'],pair=f'Rh13-Cl{cl}',distance_A=v['source_distance_A'],local_length_A=m['local_length_A'],local_angle_deg=m['local_angle_deg'],global_L=m['global_L'],global_Theta=m['global_Theta'],ICOHP_eV_per_bond=cur,delta_ICOHP_eV=cur-base,Rh4d_Cl3p_occupied_integral_eV=dp,orbital_integral_method='printed integrated orbital sum at EF' if state in remote else 'trapezoid of full occupied exported COHP; finite-grid difference retained',DFT_Eg_eV=m['Eg_eV'],delta_Eg_eV=m['delta_Eg_eV'],physical_trend_eligibility='EXCLUDE_FROM_SAME_SHORT_BOND_TREND_PENDING_IMAGE' if periodic else 'DESCRIPTIVE_ONLY_UNCERTIFIED_PROJECTION',exclusion_is_not_data_deletion=True))
  for region in ['file_identity_and_geometry','occupied_ICOHP','Rh4d_Cl3p','VBM_pm0p1eV','CBM_pm0p1eV']:
   status=('HOLD_PERIODIC_IMAGE' if periodic else 'CAN_USE_DESCRIPTIVELY') if region=='file_identity_and_geometry' else ('HOLD_PERIODIC_IMAGE' if periodic else 'HOLD_UNRESOLVED_PROJECTION')
   qc.append(dict(state=state,pair=f'Rh13-Cl{cl}',region=region,technical_status='TECHNICAL_PASS',projection_status='PROJECTION_QC_PENDING',use_status=status,limited_use_pass=False,allowed_claim='file contents and observed computed values only' if status=='CAN_USE_DESCRIPTIVELY' else 'retain as unresolved observation, no physical mechanism claim',positive_reliability_evidence='SHA/geometry/column sum/ICOHP consistency are positive INTERNAL data evidence; not independent orbital reliability proof',missing_evidence='actual internal translation/phase selection and target-projection reliability' if periodic else 'independent near-degenerate target orbital projection and sensitivity/error bound'))
  for ed in ['VBM','CBM']:
   b=edge[(state,ed)];risks.append(dict(state=state,pair=f'Rh13-Cl{cl}',edge=ed,kpoint=b['k_fractional'],band=b['band_index1'],relative_own_EF_eV=b['relative_to_own_EF_eV'],risk='A09 Gamma146 0.444meV above CBM and row deviation0.00816559, Rh13/Cl18 present; 48/48 warning occupied subblocks affected' if state=='6313886_LENGTH_06' else 'state-specific degenerate subspace/projection impact not closed; cannot transfer A09 numeric error',status='HOLD_PERIODIC_IMAGE' if periodic else 'HOLD_UNRESOLVED_PROJECTION',common_potential_reference='MISSING',DFT_response_numerical_error_bound='NOT_CERTIFIED'))
table('NINE_STATE_COHP_ICOHP_TRENDS.csv',trend);table('NINE_STATE_TARGET_PAIR_QC.csv',qc);table('BAND_EDGE_PROJECTION_RISK.csv',risks);table('BAND_EDGE_WINDOW_OBSERVATIONS.csv',windows)
write('PERIODIC_IMAGE_FINAL_ASSESSMENT.md','''
# 周期像定向结论：异常关联已定位，实际相互作用尚不能最终认证

源POSCAR中Rh13=(0,0,0.5)。ANGLE06/10 Cl19的x分别−0.0008249056111881/−0.0041243270342214，源坐标零像短键长均2.389368875586419Å。LOBSTER输出结构折回x=0.9991751/0.9958757，同一物理短键在该表示中应取(-1,0,0)；不取该平移的原胞内分离约10Å。ANGLE01/05为正x、无需跨边界；母体x=0。各结构未被修改。

冻结5.1.1随包手册LOBSTER_MANUAL_EXTRACT.txt第420–430行明确原子编号依输入顺序，并提供`cohpBetween atom 1 atom 2 cell 1 0 0`选择邻胞；第482–493行说明cohpBetween和距离生成器。实际输入仅`cohpbetween atom 13 atom 19 orbitalWise`，无cell。ICOHPLIST的零像/短距离标签与折回结构的零像几何并不构成闭合映射。原子索引相同不能证明同一物理周期键被选中。

实现核查边界：项目只有5.1.1二进制和手册，没有可审阅的该版本源码或运行时内部pair/phase索引追踪；不把手册语法冒称已读源码。已固定实际二进制SHA4ad178e33e94f7cfba96ec5f5fe946b1675b14b42561dd981491f3a3dea62e7a。官方[下载页面](https://schmeling.ac.rwth-aachen.de/cohp/index.php?menuID=6)可确认5.1.1版本及手册来源，不能替代本次运行的内部映射证据。

新读取原谱确认ANGLE06/10该键总ICOHP−0.00043/−0.00048eV；15个Rh4d–Cl3p分量在打印精度下积分和均0。Cl18通道约−4.82404/−4.83504eV，4d–3p约−0.94599/−0.94423eV。周期边界表示与异常同时出现是定位线索，不是以预期趋势判错的证据。正确标签距离、低spilling及分量加和都不能单独验证相位/平移选取。

直接回答：①尚不能证明两个近零输出就是期望的同一Rh13–Cl19短键；②真实物理响应未被逻辑排除，但现有证据不足以如此解释，也不判零键强度；③缺少5.1.1内部平移/原子相位映射或同电子态显式周期像对照；④Rh13–Cl18可以独立保留为计算描述，不受Cl19的像问题自动牵连，但仍有投影HOLD，不能授限定科学PASS；⑤当前尚不能认定必须新增LOBSTER，应先核已有诊断矩阵和内部映射。若无法离线重构，最少可从一个异常状态进行一次显式像对照试做，需独立授权。
''')
write('COHP_CLAIM_ELIGIBILITY.md','''
# 可以陈述什么
18条原子对记录、90个状态/通道/能区判定均保留，没有删除异常值。TECHNICAL_PASS与PROJECTION_QC_PENDING、SCIENTIFIC_QC_HOLD分列；CAN_USE_DESCRIPTIVELY仅指文件和算法输出的描述用途，绝不是LIMITED_USE_PASS。

可写：在冻结计算分支中，局部几何改变伴随所计算COHP/ICOHP变化；这些是待投影/像映射认证的探索性观察。Rh13–Cl18可在九态保留独立记录；七个无已知像异常状态的Cl19可记录数值。两个异常Cl19只报告“输出近零且像身份未认证”，不能并入同短键物理趋势。

不能写：几何变化已被证明导致成键/反键重排继而决定带隙；ICOHP更负就必然扩大/缩小Eg；近零表示断键；own-EF对齐表明绝对带边移动；低spilling证明目标轨道可靠。无通道达到LIMITED_USE_PASS，无证据足以判FAIL_CONFIRMED_INVALID_PAIR，不能因未见其他问题就放行。

长度Cl19观察：负端−4.87444，负中−4.84665，母体−4.78241，正中−4.81086，正端−4.75315eV，局部并非单调，不能抹平中间点。Cl18也随路径共变。角度Cl19负端/负中约−4.81537/−4.82611，而正向像未闭合。各点DFT Eg/ΔEg原值保留CSV；缺逐对数值误差界，不计算带隙因果回归或显著性。

本批针对各自采样VBM/CBM±0.1eV从真实谱计算总COHP和4d–3p窗积分（BAND_EDGE_WINDOW_OBSERVATIONS.csv），不平滑。这个窗口是描述性汇总，不是可靠性阈值。COHP为k积分且Gaussian0.05eV，无法仅从窗值归属某条k/band；尚缺完整近简并子空间影响分析，投影风险可能改变符号、排序和机制，不只是精度尾数。A09 Γ144正交性较好不认证所有占据态或Γ146。

只影响已有文件数值重构精度的已识别因素：约4–5e−5的打印分量求和差、有限网格积分差；它们不能为未知投影误差定界。共同能量参考缺失影响绝对带边移动结论；周期像未决可能改变所分析的相互作用对象。科学HOLD不是计算失败。
''')
write('MINIMUM_RECALCULATION_DECISION.md','''
# 最小补算决定：当前必需新增调用尚不能确立，先完成零调用诊断

Job13904已经执行，绝非待申请任务。本批只读目录核对现有projectionData.lobster约75.8MB及coefficientMatrices/LSO1/LSO2三文件各约62.2MB，另有原诊断输入与终态。完整目录与小JSON见REMOTE_ANALYSIS.json diagnostic。A09原“1次仅新增矩阵输出”建议已经由13904实现，不能再次作为最少新增调用提出。未重新解析A09已封存大矩阵或重算母体/L06。

优先0次：继续使用13904已存在矩阵定位L06占据与近简并子空间对目标COHP的影响；需先确认系数/相位/基组排序格式，不能把abs bandOverlaps行列当原子轨道。此次仅认证文件存在与格式来源线索，尚未完成约260MB矩阵的完整相位重构，不声称它已关闭投影门。该既有诊断不能证明ANGLE06/10的周期像。

若无源码/内部映射且不能离线重构：条件最小首试1次ANGLE06同源LOBSTER对照，保持DFT/结构/基组/展宽/能区不变，在独立目录明确打印及比较零像、期望短键像和Cl18参考。cell语法需按5.1.1实际解析核验；不直接盲填(-1,0,0)，因为输入坐标与内部折回坐标语义仍待判定。不得改POSCAR折回后直接套旧WAVECAR。先同输入一次同时选择候选像并检查实际几何，避免把投影差异与对象变化混淆。只有确证这一试做有效后，再审是否需另1次ANGLE10。额外输出参数仅在预算核验后启用；失败不自动重试。

资源仅可参考A10异常态实测：ANGLE06 48OMP约475.5s/5.323GiB，ANGLE10 mgr40OMP约605.6s/5.335GiB；诊断像/轨道输出扩张后的耗时、磁盘和内存UNKNOWN，不能承诺等同。节点由另批实时资源核对，当前无新授权。新增DFT需求0；新增LOBSTER当前执行0，条件首次1、第二点另审1，不保证两次即可科学通过。

投影基组敏感性只在13904现有证据仍不能验证具体主张时另评，不把更换基组当默认修复，不加NBANDS，不先补算九态。若作者仅采用明确局限性的探索性描述，当前无需为画图或趋势展示重复计算。
''')
write('FINAL_SCIENTIFIC_ACCEPTANCE.md','''
# A13最终分级：技术九态通过；科学限定PASS暂为0

完成九态18条目标原子对、五个用途/能区共90项分级，见NINE_STATE_TARGET_PAIR_QC.csv。所有态TECHNICAL_PASS；投影PROJECTION_QC_PENDING；目标机制总体SCIENTIFIC_QC_HOLD。LIMITED_USE_PASS0、SCIENTIFIC_FAIL0，不因0.94%spill放行，也不因正交化警告判全部失效。

1. 可用通道：九态Rh13–Cl18与除ANGLE06/10外七态Rh13–Cl19可作为文件数值的描述性观察；两异常Cl19只保留异常记录。所有占据ICOHP、4d–3p及带边物理归因仍HOLD，未获正式限定科学用途资格。
2. ANGLE06/10近零原因未最终明确；负坐标折回、无显式cell、零像标签和4d–3p零积分构成待核映射证据，不能凭趋势认定程序错误或真实断键。
3. 打印舍入、已量化的网格积分差主要影响重构数值精度。它们不是投影误差的上界。
4. 周期像/相位身份、占据与近简并轨道投影不稳、DFT ΔEg缺误差界、无共同参考势均可能改变物理解释，不能统称精度小问题。
5. 可以支持带明确计算及投影局限性的探索性定性描述“局部几何与计算成键指标共变”；尚不能作为已验证的短键响应方向、单调规律或带隙因果机制。正式论文若要这些更强主张需补证据，而非直接删异常。
6. 尚未证明必须新算。先0次使用Job13904已有矩阵和映射；必要时一个异常态1次显式像同源LOBSTER试做，之后另审第二态。DFT0；不重复已完成诊断输出任务。

本批真实读取七态COHP只为新轨道/能窗汇总，SHA与run106输出证据一致；PARENT/L06使用A11已验证CSV与A09结果，不重提取大文件。读取Job13904文件清单和现有小JSON，无计算。未修改原谱、图件、科学原QC或正式重要性矩阵。身份、语法文档、实际运行行为三者严格区分：源码/内部pair相位追踪未取得，未声称审计了闭源实现。

交付已完成，内部映射及完整诊断系数的科学问题未决，下一步据本批最小方案由作者决定；本批停止。项目整体未完成。
''')
write('CHECKPOINT.md','''
# run107 COHP A13：九态科学分级与最小周期像诊断
状态SEALED_DIRECTIONAL_ANALYSIS_COMPLETE_WITH_SCIENTIFIC_HOLDS。九态技术PASS；18对/90用途判定；科学限定PASS0、确认FAIL0。16对可记录描述性数值（对象/文件层），两正角Cl19 HOLD_PERIODIC_IMAGE，所有物理轨道/带边机制未获科学PASS。真实七态4d–3p及VBM/CBM±0.1eV积分完成，未新增绘图。
Job13904已有projectionData及三套系数文件；不能重复申请“首次诊断输出”。当前新增调用必要性未证明，优先0调用离线矩阵/映射分析，条件首试1次异常态显式像LOBSTER、另一态另审。连接通过用户本轮交互认证恢复；密码未写入任何文件。不重跑、不修改远端文件或原图；新DFT/LOBSTER/提交0。
继续只读FINAL_SCIENTIFIC_ACCEPTANCE.md、PERIODIC_IMAGE_FINAL_ASSESSMENT.md、MINIMUM_RECALCULATION_DECISION.md和CSV。下一阶段需作者指定，不自动计算。旧run106/A09/A11保留，项目未完成。
''')
write('NEXT_BATCH_INSTRUCTIONS.md','优先零新增调用审查13904现有系数/相位和5.1.1内部像定义；未完成前不得给LIMITED_USE_PASS。若须实算先冻结单异常态最小对照和预算，另行授权。本批不修改旧值/标签/图件，不自动递补。')
inputs=[_repo_path('08_COHP_ANALYSIS/workflows/projection_checks/INDEPENDENT_OUTPUT_AUDIT_FINAL.json'),_repo_path('08_COHP_ANALYSIS/workflows/projection_checks/TARGET_PAIR_ICOHP_AUDIT.csv'),_repo_path('08_COHP_ANALYSIS/workflows/projection_checks/ANGLE_COORDINATE_REPRESENTATION_AUDIT.json'),_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/NINE_STATE_STRUCTURE_AUDIT.csv'),_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/BAND_EDGE_RECOVERY.csv'),_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/LOBSTER_MANUAL_EXTRACT.txt'),_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/FINAL_ACCEPTANCE_DECISION.md'),_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/NUMERICAL_SUMMARY.json'),_repo_path('unavailable_inputs/source_0796008550b2/EXISTING_COMPLETION_STATES.json'),_repo_path('08_COHP_ANALYSIS/workflows/original_spectra/TRIAL_SUMMARY.json')]
table('SOURCE_SHA256.csv',[dict(path=str(p),sha256=sha(p)) for p in inputs]);write('VALIDATION.json',json.dumps(dict(states=9,pairs=18,scoped_decisions=len(qc),window_summaries=len(windows),limited_use_pass=0,confirmed_invalid=0,all_finite_window_values=all(np.isfinite(x['COHP_window_integral']) for x in windows),new_DFT=0,new_LOBSTER=0,plots_modified=0),indent=2))
print('A13 reports complete',len(trend),len(qc),len(windows))
