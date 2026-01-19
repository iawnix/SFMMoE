import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdMolTransforms import GetAngleDeg
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
from torch_geometric.data import Data, InMemoryDataset
from rich.progress import track
from typing import Dict, List, Set
from types import FunctionType
from  numpy import ndarray as arr

def cal_mol_sele_feat(mol):
    descriptor_names = ['MinAbsEStateIndex', 'MinEStateIndex', 'SPS', 'FpDensityMorgan1'
                        , 'AvgIpc', 'Chi4n', 'Kappa3', 'PEOE_VSA1', 'PEOE_VSA6', 'SMR_VSA10', 'SMR_VSA2'
                        , 'SMR_VSA3', 'SlogP_VSA11', 'SlogP_VSA4', 'SlogP_VSA7', 'SlogP_VSA8', 'EState_VSA2'
                        , 'EState_VSA4', 'EState_VSA5', 'EState_VSA9', 'VSA_EState9', 'NHOHCount', 'NumAliphaticCarbocycles'
                        , 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings'
                        , 'NumHDonors', 'RingCount']
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    return list(descriptor_calculator.CalcDescriptors(mol))

class Pdb2PYG():
    def __init__(self, mol_id: str = None ,mol = None, fp_pdb: str = None, fp_mol: str =None
                 , fp_charges: str = None, fp_who: str = None
                 , iaw_feat: bool = False, qm_feat: List = None) -> None:
        self.more_feat = False
        self.mol_id = mol_id
        self.iaw_feat = iaw_feat
        self.qm_feat = qm_feat

        self.pt = Chem.GetPeriodicTable()

        if fp_pdb:
            self.mol = Chem.MolFromPDBFile(fp_pdb, removeHs=False)
            self.more_feat = True
        elif fp_mol:
            self.mol = Chem.MolFromMolFile(fp_mol, removeHs=False)
            self.more_feat = True
        elif mol:
            self.mol = mol
        else:
            # fatal error
            pass

        if not fp_charges:
            # fatal error
            pass
        else:
            self.charges: List[float] = self.__read_charge__(fp_charges)
        # 计算原子总数
        atm_n = self.mol.GetNumAtoms()
        self.wbo, self.__min__wbo, self.__max__wbo = self.__read_wbo__(fp_who, atm_n)
        
        # 这里将自定义的参数加入mol中
        ## charges
        for atm in self.mol.GetAtoms():
            atm.SetProp("ESP",str(self.charges[atm.GetIdx()]))
        ## wbo
        for bond in self.mol.GetBonds():
            atm1_id = bond.GetBeginAtomIdx()
            atm2_id = bond.GetEndAtomIdx()
            _wbo = self.wbo[atm1_id, atm2_id]
            bond.SetProp("WBO", str(_wbo))
        
        # 这里去除氢
        self.mol = Chem.RemoveAllHs(self.mol)
        self.conf = self.mol.GetConformer()

    def __read_charge__(self, fp) -> List[float]:
        """
        File Format: 
            num1
            num1
            ...
        """
        out = []
        with open(fp, "r+") as F:
            for line in F.readlines():
                out.append(eval(line.rstrip("\n")))
        return out

    def __read_wbo__(self, fp, atm_n) ->List[List[float]]:
        """
        File Format: 
            atm1 atm2 wbo
            ...
        """
        out = np.zeros(shape=(atm_n, atm_n))
        minmax = []
        with open(fp, "r+") as F:
            for line in F.readlines():
                
                line1 = line.rstrip("\n")
                #line1 = line1.replace("\t", " ").replace("  ", " ")
                var = line1.split(" ")
                var = [i for i in var if i != ""]
         
                atm1, atm2, wbo = var[0], var[1], var[2]
                atm1 = eval(atm1)
                atm2 = eval(atm2)
                wbo = eval(wbo)
                out[atm1-1, atm2-1] = wbo
                out[atm2-1, atm1-1] = wbo
                minmax.append(wbo)

        min_ = np.array(minmax).min()
        max_ = np.array(minmax).max()
        return out, min_, max_
 
    def atm_f(self, atm) -> arr:
        """
            对原子进行编码
        """
        f_type: Dict = {
              "atm_sym": [i.upper() for i in ["S", "F", "Br", "Cl", "P",	"I", "O", "N", "Se", "Si", "C", "B", "H", "undefined"]]
            , "atm_dgre": [0, 1, 2, 3, 4, "MoreThanFour"]
            , "atm_Hn":[0,1,2,3,4, "MoreThanFour"]
            , "atm_chag": lambda x: x
            , "atm_mass": lambda x: x
            , "atm_vdw":  lambda x: x
            , "atm_Hbz":  ["SP", "SP2", "SP3", "SP3D","SP3D2","UNSPECIFIED"]
            , "atm_IV":  [0,1,2,3,4, "MoreThanFour"]
            , "atm_EV":  [0,1,2,3,4, "MoreThanFour"]
        }

        # 元素
        # 这里是bug

        atm_sym = self.encode_node(f_type["atm_sym"], atm.GetSymbol().upper())
        # 度
        atm_dgre = self.encode_node(f_type["atm_dgre"], atm.GetDegree())
        # 氢原子数目
        atm_Hn = self.encode_node(f_type["atm_Hn"], atm.GetTotalNumHs())
        # 电荷
        atm_chag = self.encode_node(f_type["atm_chag"], eval(atm.GetProp("ESP")))
        # 质量
        atm_mass = self.encode_node(f_type["atm_mass"], atm.GetMass())
        # vdw
        atm_vdw = self.encode_node(f_type["atm_vdw"], self.pt.GetRvdw(atm.GetSymbol()))
        # 判断是否环上
        atm_in_ring = [int(atm.IsInRing())]
        if self.more_feat:
            # 原子的杂化类型
            atm_Hbz = self.encode_node(f_type["atm_Hbz"], str(atm.GetHybridization()).upper())
            # 原子的隐价态
            atm_IV = self.encode_node(f_type["atm_IV"], atm.GetImplicitValence())
            # 原子的显价态
            atm_EV = self.encode_node(f_type["atm_EV"], atm.GetExplicitValence())
            # 是否为芳香性原子
            atm_Iarom = [int(atm.GetIsAromatic())]
            atm_f_v = atm_sym+atm_dgre+atm_Hn+atm_chag+atm_mass+atm_vdw+atm_in_ring +atm_Hbz + atm_IV + atm_EV + atm_Iarom
        else:
            atm_f_v = atm_sym+atm_dgre+atm_Hn+atm_chag+atm_mass+atm_vdw+atm_in_ring
        return np.array(atm_f_v)

    def encode_node(self, f_name, atm_value) -> List:
        if type(f_name) == list:
            if atm_value not in f_name:
                atm_value = f_name[-1]
            return [int(boolean_value) for boolean_value in list(map(lambda s: atm_value == s, f_name))]
        if type(f_name) == FunctionType:
            return [f_name(atm_value)]
    
    def bond_f(self, bond) -> arr:
        atm1 = bond.GetBeginAtom()
        atm2 = bond.GetEndAtom()
        
        # 键长
        cal_bond = lambda atm1, atm2: np.linalg.norm( self.conf.GetAtomPosition(atm1.GetIdx()) - self.conf.GetAtomPosition(atm2.GetIdx()))
        bond_len = [cal_bond(atm1, atm2)]

        # Wiberg Bond Order (WBO键级)
        bond_wbo = [eval(bond.GetProp("WBO"))]

        # 判断是否环上
        bond_in_ring = [int(bond.IsInRing())]
        
        if self.more_feat:
            bond_type = [int(boolean_value) for boolean_value in list(map(lambda s: str(bond.GetBondType()).upper() == s, ["SINGLE", "DOUBLE", "AROMATIC", "TRIPLE"]))]
            # 去除键长
            bond_f_v = bond_len + bond_wbo+bond_in_ring + bond_type
            #bond_f_v = bond_wbo+bond_in_ring + bond_type
        else:
            bond_f_v = bond_len + bond_wbo+bond_in_ring
            #bond_f_v =  bond_wbo+bond_in_ring
        return np.array(bond_f_v)
    
    def ang_f(self, ang_i_j_k) -> arr:
        i, j, k = ang_i_j_k 
        # 这里只计算角度
        return [GetAngleDeg(self.conf, i, j, k)] 
    
    def construct_ang_index(self):
        ang_index = []
        for atm in self.mol.GetAtoms():
            neighbors = [neighbor.GetIdx() for neighbor in atm.GetNeighbors()]
            if len(neighbors) < 2:
                continue
            for i in range(len(neighbors)):
                for k in range(i + 1, len(neighbors)):
                    ang_index.append([neighbors[i], atm.GetIdx(), neighbors[k]])
                    ang_index.append([neighbors[k], atm.GetIdx(), neighbors[i]])
        
        # [3, n_angles]
        return torch.tensor(ang_index).T
    
    def construct_edge_index(self):
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(self.mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        
        # [2, n_edges]
        return torch.stack([torch_rows, torch_cols], dim = 0)

    def index_edge_i_j(self, edge_index, i, j):
        row = torch.where((edge_index[0,:] == i))[0]
        col = torch.where((edge_index[1,:] == j))[0]
        return row[torch.isin(row,col)]
    
    def index_edge_is_js(self, edge_index, i_s, j_s):
        out = []
        for _, i in enumerate(i_s):
            out.append(self.index_edge_i_j(edge_index, i, j_s[_]))
        return torch.cat(out, dim=0)


    def __call__(self, y:List) -> arr:
    
        # node 特征
        # [n_nodes, n_node_features]
        n_node_features = len(self.atm_f(self.mol.GetAtomWithIdx(0)))
        n_nodes = self.mol.GetNumAtoms()
        X = np.zeros((n_nodes, n_node_features))
        for atm in self.mol.GetAtoms():
            X[atm.GetIdx(), :] = self.atm_f(atm)
        X = torch.tensor(X, dtype = torch.float)

        # edge 特征
        # [n_edges, n_edge_attrs]
        n_edge_features = len(self.bond_f(self.mol.GetBonds()[0]))  
        n_edges = 2*self.mol.GetNumBonds()  
        edge_attr = np.zeros((n_edges, n_edge_features))
        edge_index = self.construct_edge_index()
        for _idx in range(edge_index.size()[1]):
            i = edge_index[0, _idx].item()
            j = edge_index[1, _idx].item()
            edge_attr[_idx] = self.bond_f(self.mol.GetBondBetweenAtoms(int(i),int(j)))
        edge_attr = torch.tensor(edge_attr, dtype = torch.float)

        # angle 特征
        ang_index = self.construct_ang_index()
        n_angles = ang_index.size()[1]
        n_angles_features = len(self.ang_f(ang_index[:,1].tolist()))  
        ang_attr = np.zeros((n_angles, n_angles_features))
        for i in range(n_angles):
            ang_attr[i] = self.ang_f(ang_index[:,i].tolist())
        ang_attr = torch.tensor(ang_attr, dtype = torch.float)

        # angle_edge_attr
        i_ = ang_index[0,:]
        h_ = ang_index[1,:]
        j_ = ang_index[2,:]
        h_i = edge_attr[self.index_edge_is_js(edge_index, h_, i_)]
        j_h = edge_attr[self.index_edge_is_js(edge_index, j_, h_)]
        angle_edge_attr = torch.stack([h_i, j_h], dim=1)
        
        # 预测值
        y_tensor = torch.tensor(np.array(y), dtype = torch.float)

        # 坐标信息
        pos = []
        for i in range(self.mol.GetNumAtoms()):
            var = self.conf.GetAtomPosition(i)
            pos.append([var.x, var.y, var.z])
        pos = torch.tensor(np.array(pos), dtype = torch.float)

        iaw_feat_ = []
        if self.iaw_feat:
            iaw_feat_ += cal_mol_sele_feat(self.mol)
            #print(iaw_feat, cal_mol_sele_feat(self.mol))
        if self.qm_feat:
            iaw_feat_ += self.qm_feat            
        iaw_feat_tensor = torch.tensor(([iaw_feat_]), dtype = torch.float)

        return Data(x = X
                        , edge_index = edge_index, edge_attr = edge_attr
                        , angle_index = ang_index, angle_attr = ang_attr, angle_edge_attr = angle_edge_attr
                        , pos = pos
                        , y = y_tensor
                        , iaw_attr = iaw_feat_tensor
                        , iaw_id = self.mol_id)

torch.serialization.add_safe_globals({'Pdb2PYG': Pdb2PYG})

