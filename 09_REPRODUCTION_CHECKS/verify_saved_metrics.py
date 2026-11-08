"""Deterministic checks of saved rows; no fitting, inference or resampling."""
from pathlib import Path
import csv,json,math,gzip,collections
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 allrows=rows(ROOT/'07_DFT_VALIDATION/data/saved_statistics/FIG6E_NODE_DIAGNOSTICS_ALL_440.csv');r=[x for x in allrows if x['primary_distribution'].lower()=='true']
 assert len(allrows)==440 and len(r)==371 and len({x['material_id'] for x in r})==17
 assert sum(x['outside_local_box_any'].lower()=='false' for x in r)==124
 y=[float(x['delta_Eg_eV']) for x in r];p=[float(x['CRISP_X_predicted_delta_eV']) for x in r];et=[float(x['ET_direct_delta_eV']) for x in r]
 n=len(y);m=sum(y)/n;err=[a-b for a,b in zip(p,y)];mae=sum(abs(x) for x in err)/n;rmse=math.sqrt(sum(x*x for x in err)/n);r2=1-sum(x*x for x in err)/sum((x-m)**2 for x in y)
 direction=sum((a>0)-(a<0)==(b>0)-(b<0) for a,b in zip(p,y))/n
 majority=max(sum(x>0 for x in y),sum(x<0 for x in y),sum(x==0 for x in y))/n
 assert abs(mae-.004002608318108136)<1e-12 and abs(r2+4.139611421605109)<1e-9 and abs(direction-.5902964959568733)<1e-12
 parameters=rows(ROOT/'07_DFT_VALIDATION/data/supplementary/SUPPLEMENTARY_DFT_CALCULATION_PARAMETERS.csv');diff=rows(ROOT/'07_DFT_VALIDATION/data/supplementary/SUPPLEMENTARY_DFT_AB_DIFFERENCES.csv')
 assert len(parameters)==1575 and len(diff)==48
 features=rows(ROOT/'03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv');folds=rows(ROOT/'03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv')
 assert len(features)==len(folds)==7176 and {x['ID'] for x in features}=={x['ID'] for x in folds}
 assert collections.Counter(x['outer_fold'] for x in folds)['0']==1462
 groups={}
 for x in folds:groups.setdefault(x['formula_group'],set()).add(x['outer_fold'])
 assert all(len(x)==1 for x in groups.values())
 ids=json.loads((ROOT/'02_ALIGNN_TRAINING/data/splits/ids_train_val_test.json').read_text());assert [len(ids[k]) for k in ['id_train','id_val','id_test']]==[4521,645,1291]
 assert not (set(ids['id_train'])&set(ids['id_test']) or set(ids['id_val'])&set(ids['id_test']) or set(ids['id_train'])&set(ids['id_val']))
 oq24=rows(ROOT/'05_CRISP_X/data/local_reconstruction/new_parent_manifest.csv');j38=rows(ROOT/'06_EXTERNAL_TRANSFER/data/external_response/PILOT38.csv');assert len(oq24)==24 and len(j38)==38
 gp=rows(ROOT/'07_DFT_VALIDATION/data/saved_statistics/SI_DFT_F6-T22_F6-T22_DFT_Scores_Source_And_Uncertainty.csv');gp=[x for x in gp if x['method']=='DFT geometry perturbation' and x['n_observations']!='NA'];assert len(gp)==2 and all(float(x['n_parents'])==16 and float(x['n_observations'])==349 for x in gp)
 assert len({x['record_id'] for x in parameters})==22 and len({(x['record_id'],x['role']) for x in parameters})==44 and len({x['record_id'] for x in parameters if x['evidence_group']=='FORMAL'})==13
 raw=ROOT/'01_DATA_CURATION/data/raw/DPVK-OQMD-cif.csv.gz'
 with gzip.open(raw,'rt',encoding='utf-8-sig',newline='') as f:source_count=sum(1 for _ in csv.DictReader(f))
 assert source_count==72747
 print(json.dumps({'source_materials':source_count,'final_materials':7176,'fold0_heldout':1462,'formula_groups_cross_fold':0,'ALIGNN_internal_train_val_test':[4521,645,1291],'OQ24_parents':24,'J38_parents':38,'DFT_GP_nodes':349,'DFT_GP_parents':16,'S14_formal_pairs':13,'S14_all_pair_records':22,'S14_role_endpoints':44},indent=2))
 print(json.dumps({'nodes':n,'parents':17,'in_range':124,'out_of_range':247,'CRISP_X_MAE_eV':mae,'CRISP_X_RMSE_eV':rmse,'CRISP_X_R2':r2,'direction_agreement':direction,'majority_direction_baseline':majority,'ET_MAE_eV':sum(abs(a-b) for a,b in zip(et,y))/n,'zero_response_MAE_eV':sum(abs(x) for x in y)/n,'S14_recorded_parameters':len(parameters),'S14_AB_differences':len(diff),'science_rerun':False},indent=2))
if __name__=='__main__':main()
