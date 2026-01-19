

# 读入mol之后需要进行pyg data的制作
class mol_2_pyg_data():
    def __init__(self, mol):
        self.mol = mol
        scratch = "/home/iaw/MYscrip/WEB/SFMMoE/.cache"
        self.job_id = uuid.uuid4()

        self.work_home = os.path.join(scratch, "{}-{}".format(datetime.date.today(), str(self.job_id)))
        os.system("mkdir {}".format(self.work_home))

        self.__to__pdb__()
    

    def __to__pdb__(self):
        if os.path.exists(self.work_home):
            os.chdir(self.work_home)
            Chem.MolToPDBFile(self.mol, "mol_for_xtb.pdb")

    def __get_more_gap__(self, fp):
        sign1 = False
        sign2 = False
        dat = []
        with open(fp , "r+") as F:
            Occ = True
            for line in F.readlines():
                if "Orbital Energies and Occupations" in line.rstrip("\n"):
                    sign1 = True
                if "-------------------------------------------------------------" in line.rstrip("\n"):
                    if sign1 and sign2:
                        break
                    else:
                        sign2 = not sign2
                        continue
                if sign1 and sign2:
                    var = [ss for ss in line.rstrip("\n").split(" ") if (ss != "") and (ss != "...")]
                    if len(var) == 0:
                        continue
                    
                    if Occ:
                        if "(HOMO)" not in var and "(LUMO)" not in var:
                            var.append("(HOMO)")
                        elif "(HOMO)" in var:
                            pass
                        elif "(LUMO)" in var:
                            if len(var) == 4:
                                var.insert(1, "0.0000")
                            Occ = False
                    else:
                        if "(LUMO)" not in var:
                            if len(var) == 3:
                                var.insert(1, "0.0000")
                            var.append("(LUMO)")

                    dat.append(var)
                #else:
                #    print(sign1, sign2, line)
        return dat
    
    def __more_gap_format__(self, dat, num):
        dat = [[eval(lis[0]), eval(lis[1]), eval(lis[2]), eval(lis[3]), lis[4].replace("(", "").replace(")", "")] for lis in dat]

        last_homo_idx = None
        first_lumo_idx = None

        for i, i_t in enumerate(dat):
            if i_t[-1] == 'HOMO':
                last_homo_idx = i  
            if i_t[-1] == 'LUMO' and first_lumo_idx is None:
                first_lumo_idx = i 
        #print(last_homo_idx, first_lumo_idx)

        homo = [i[-2] for i in dat[last_homo_idx-num+1:last_homo_idx+1]]
        homo.reverse()
        lumo = [i[-2] for i in dat[first_lumo_idx:last_homo_idx+num+1]]
        homo_label = list(range(len(homo)))
        lumo_label = list(range(len(lumo)))
        #if len(homo_label) + len(lumo_label) != 2*num:
        #    return -1
        out = {}
        for i, j in list(itertools.product(homo_label, lumo_label)):
            out["HOME_{}-LUMO_{}".format(i, j)] = homo[i] - lumo[j]
        return out

    def __xtb__(self):

        os.chdir(self.work_home)
        i_charge = Chem.rdmolops.GetFormalCharge(self.mol)
        

        # 补丁, 增加对自旋的判定
        chrg = Chem.GetFormalCharge(self.mol)                # 计算分子的净电荷
        re = Descriptors.NumRadicalElectrons(self.mol)       # 计算分子所具有的自由基电子数
        ve = Descriptors.NumValenceElectrons(self.mol)       # 计算分子的价电子数
        # 检查总电子数是否为偶数且没有自由基电子
        if (ve + chrg) % 2 == 0 and re == 0:
            uhf = 0
        else:
            uhf = re

        out1 = CMD_RUN("/opt/xtb/6.7.1/bin/xtb ./mol_for_xtb.pdb --ohess --chrg {} --uhf {} --opt normal --molden > xtb.log".format(i_charge, uhf))

        # 补丁 这里需要移动一下charges文件
        CMD_RUN("mv charges old_charges")
        CMD_RUN("mv charges old_charges")
        CMD_RUN("mv wbo old_wbo")
        out2 = CMD_RUN("/opt/xtb/6.7.1/bin/xtb ./xtbopt.pdb --vipea > Vipea.log")
        CMD_RUN("mv charges new_charges")
        CMD_RUN("mv old_charges charges")
        CMD_RUN("mv wbo new_wbo")
        CMD_RUN("mv old_wbo wbo")

        os.system("grep -v 'END'  ./xtbopt.pdb > final.pdb")
        os.system("grep 'CONECT' mol_for_xtb.pdb >> final.pdb")
        
        final_pdb_fp = os.path.join(self.work_home, "final.pdb")
        final_charges_fp = os.path.join(self.work_home, "charges")
        final_wbo_fp = os.path.join(self.work_home, "wbo")
        final_log_fp = os.path.join(self.work_home, "Vipea.log")
        final_hl_log = os.path.join(self.work_home, "xtb.log")
        # final.pdb: xyz
        # wbo
        s1_, ip_, e_ = CMD_RUN("grep 'delta SCC IP (eV):' {}".format(final_log_fp))
        s2_, ea_, e_ = CMD_RUN("grep 'delta SCC EA (eV):' {}".format(final_log_fp))
        s3_, hl_, e_ = CMD_RUN("grep 'HOMO-LUMO GAP' {}".format(final_hl_log))
        print(ip_)
        print(ea_)
        _gap = self.__get_more_gap__(final_hl_log)
        
        new_homo_lumo_4 = molden_mol(fp = "./molden.input")
        # 这里存在bug
        # 需要返回到WEB_HOMOE这个地方

        if (s1_ == 1) and (s2_ == 1) and (s3_ == 1) and len(_gap) >= 4:
            gap = self.__more_gap_format__(_gap, 4)
            return (final_pdb_fp
                        , final_charges_fp
                        , final_wbo_fp
                        , ip_.rstrip("\n").replace("delta SCC IP (eV):", "").replace(" ","")
                        , ea_.rstrip("\n").replace("delta SCC EA (eV):", "").replace(" ","")
                        , gap, new_homo_lumo_4.print_FO_streamlit())
        else:
            return -1
    
    def __call__(self):
        out = self.__xtb__()
        if type(out) != int:
            final_pdb_fp = out[0] 
            final_charges_fp = out[1]
            final_wbo_fp = out[2]
            ip = out[3]
            ea = out[4]
            gap = out[5]
            hl_4 = out[6]
            print(ip, ea, type(ip), type(ea))
            _gaps = [eval(ip), eval(ea)]
            
            for iaw in ['HOME_0-LUMO_0', 'HOME_0-LUMO_1', 'HOME_0-LUMO_2', 'HOME_0-LUMO_3'
                             , 'HOME_1-LUMO_0', 'HOME_1-LUMO_1', 'HOME_1-LUMO_2', 'HOME_1-LUMO_3'
                             , 'HOME_2-LUMO_0', 'HOME_2-LUMO_1', 'HOME_2-LUMO_2', 'HOME_2-LUMO_3'
                             , 'HOME_3-LUMO_0', 'HOME_3-LUMO_1', 'HOME_3-LUMO_2', 'HOME_3-LUMO_3']:
                _gaps.append(gap[iaw])

            pt = Pdb2PYG(mol_id= str(self.job_id ), mol = None
                         , fp_pdb = final_pdb_fp, fp_mol = None, fp_charges = final_charges_fp, fp_who = final_wbo_fp
                         , iaw_feat = False
                         , qm_feat = _gaps)([])

             # 保存
            
            torch.save(pt, final_pdb_fp.replace(".pdb", ".pt"))

           
            return pt, hl_4

