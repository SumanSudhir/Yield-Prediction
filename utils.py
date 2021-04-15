import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats



def processGndData(filepath):
    df  = pd.read_excel(filepath)
    df = df[df.columns[[1,2,3]]]
    LB = df['Leaf Biomass (Kg/ha)'].values
    SB = df['Stem Biomass (Kg/ha)'].values
    LAI = df['LAI'].values
    reg = LinearRegression().fit(LB.reshape(-1, 1), SB)
    m = reg.coef_[0]
    b = reg.intercept_
    SLA = []
    for i in range(len(LAI)):
        SLA.append(LAI[i]/LB[i])

    SB_pred = reg.predict(LB.reshape(-1, 1))

    df_new = pd.DataFrame()
    df_new['Leaf_Biomass'] = LB
    df_new['Stem_Biomass'] = SB
    df_new['LAI'] = LAI
    df_new['Pred_Stem_Biomass'] = SB_pred
    df_new['SLA'] = SLA

    mean = np.mean(SLA)
    std = np.std(SLA)

    return df_new

def processDroneData(df, filepath):
    SLA = df['SLA']
    mean = np.mean(SLA)
    std = np.std(SLA)

    dfd = pd.read_excel(filepath)
    n_plots = dfd.shape[0]
    n_days = dfd.shape[1]-1
    dfd_new = pd.DataFrame()
    for i in range(n_days):
        dfd_new[f'LAI-{i+1}'] = dfd[dfd.columns[i+1]]
    for j in range(n_days):
        lai = dfd_new[dfd_new.columns[j]]
        lai_mu = lai.mean()
        lai_std = lai.std()
        SLA_new = []
        LB_new = []
        SB_new = []
        AGB_new = []
        for i in range(n_plots):
            if(lai[i] < lai_mu-lai_std):
                SLA_new.append(mean-std)
            elif(lai[i] > lai_mu+lai_std):
                SLA_new.append(mean-std)
            else:
                SLA_new.append(mean)

            LB_new.append(lai[j]/SLA_new[i])
            SB_new.append(0.73257658*LB_new[i] + 344.603599822683)
            AGB_new.append(LB_new[i] + SB_new[i])


        dfd_new[f'PRED_SLA-{j+1}'] = SLA_new
        dfd_new[f'PRED_LB-{j+1}'] = LB_new
        dfd_new[f'PRED_SB-{j+1}'] = SB_new
        dfd_new[f'PRED_AGB-{j+1}'] = AGB_new

    days_gap = [21-12, 12-21+30, 19-12]
    N = 53
    col = []
    for i in range(n_days-1):
        dfd_new[f'AGB_Rate-{i+1}-{i+2}'] = (dfd_new[f'PRED_AGB-{i+2}'] - dfd_new[f'PRED_AGB-{i+1}'])/days_gap[i]
        col.append(f'AGB_Rate-{i+1}-{i+2}')

    dfd_new['MeanAGBRate'] = dfd_new[col].mean(1)
    dfd_new['AGB_maturity'] = dfd_new[f'PRED_AGB-{n_days}'] + dfd_new['MeanAGBRate']*N
    dfd_new['Crop_Yield'] = 0.48*dfd_new['AGB_maturity']

    return dfd_new, n_days, n_plots

def save_LBPLot(df_new, path):
    f= plt.figure(figsize=(12,4))
    ax=f.add_subplot(121)
    sns.distplot(df_new['Leaf_Biomass'],bins=50,color='r',ax=ax)
    ax.set_title('Distribution of Leaf Biomass (Kg/ha)')
    ax=f.add_subplot(122)
    sns.distplot(np.log10(df_new['Leaf_Biomass']),bins=40,color='b',ax=ax)
    ax.set_title('Distribution of Leaf Biomass (Kg/ha) in $log$ sacle')
    ax.set_xscale('log')
    plt.savefig(path, dpi=100)


def save_RegPlot(df_new, path):
    f= plt.figure(figsize=(12,4))
    plt.scatter(df_new['Leaf_Biomass'], df_new['Stem_Biomass'],color='b')
    plt.plot(df_new['Leaf_Biomass'], df_new['Pred_Stem_Biomass'],color='r')
    plt.xlabel('Leaf Biomass (Kg/ha)')
    plt.ylabel('Stem Biomass (Kg/ha)')
    plt.savefig(path, dpi=100)

def save_CompReg(df_new, path):
    f= plt.figure(figsize=(12,4))
    plt.scatter(range(len(df_new['Leaf_Biomass'])), df_new['Stem_Biomass'], marker='x', color='r', label='Stem Biomass')
    plt.scatter(range(len(df_new['Leaf_Biomass'])), df_new['Pred_Stem_Biomass'], marker='o', color='b', label='Pred Stem Biomass')
    plt.xlabel('Plot Number (Kg/ha)')
    plt.ylabel('Stem Biomass (Kg/ha)')
    plt.title('Calculated and Given Stem Biomass')
    plt.legend(loc='best', fontsize=7)
    plt.savefig(path, dpi=100)

def save_SLAGauss(df_new, path):
    f= plt.figure(figsize=(12,4))
    mean = np.mean(df_new['SLA'])
    std = np.std(df_new['SLA'])
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    plt.plot(x, stats.norm.pdf(x, mean, std))
    plt.axvline(x=mean+std, color="red")
    plt.axvline(x=mean-std, color="red")
    plt.axvline(x=mean, color="green")
    # plt.fill_between(x, y1, 1)
    # plt.fill_between(x,stats.norm.pdf(x, mean, std))
    plt.scatter(df_new['SLA'], stats.norm.pdf(df_new['SLA'], mean, std),  color='r')
    plt.text(mean+std, 0, 'SLA mean+std', fontsize=8)
    plt.text(mean-1.5*std, 0, 'SLA mean-std', fontsize=8)
    plt.text(mean, 0, 'SLA mean', fontsize=8)

    plt.title('SLA Points distribution')
    plt.savefig(path, dpi=100)

def save_DScatter(dfd_new, path,n_days, n_plots ):
    f= plt.figure(figsize=(12,4))
    markers = ['x', '*', '^', 'o']
    colours = ['b', 'g', 'black', 'red']

    for i in range(n_days):
        plt.scatter(range(len(dfd_new)), dfd_new[f'LAI-{i+1}'], marker=markers[i%4], color=colours[i%4], label=f'LAI-{i+1}')

    plt.legend(loc='best', fontsize=7)
    plt.xlabel('Plot Number')
    plt.ylabel('LAI')
    plt.title('LAI of different dates')
    plt.savefig(path, dpi=100)

def save_DLBScatter(dfd_new, path,n_days, n_plots):
    f= plt.figure(figsize=(12,4))
    markers = ['x', '*', '^', 'o']
    colours = ['b', 'g', 'black', 'red']

    for i in range(n_days):
        plt.scatter(range(len(dfd_new)), dfd_new[f'PRED_LB-{i+1}'], marker=markers[i%4], color=colours[i%4], label=f'PRED_LB-{i+1}')

    plt.legend(loc='best', fontsize=7)
    plt.xlabel('Plot Number')
    plt.ylabel('Leaf Biomass')
    plt.title('Calculated Leaf Biomass of different dates')
    plt.savefig(path, dpi=100)

def save_DSBScatter(dfd_new, path,n_days, n_plots):
    f= plt.figure(figsize=(12,4))
    markers = ['x', '*', '^', 'o']
    colours = ['b', 'g', 'black', 'red']

    for i in range(n_days):
        plt.scatter(range(len(dfd_new)), dfd_new[f'PRED_SB-{i+1}'], marker=markers[i%4], color=colours[i%4], label=f'PRED_SB-{i+1}')

    plt.legend(loc='best', fontsize=7)
    plt.xlabel('Plot Number')
    plt.ylabel('Stem Biomass')
    plt.title('Calculated Stem Biomass of different dates')
    plt.savefig(path, dpi=100)


def save_DSLAScatter(dfd_new, path,n_days, n_plots):
    f= plt.figure(figsize=(12,4))
    markers = ['x', '*', '^', 'o']
    colours = ['b', 'g', 'black', 'red']

    for i in range(n_days):
        plt.scatter(range(len(dfd_new)), dfd_new[f'PRED_SLA-{i+1}'], marker=markers[i%4], color=colours[i%4], label=f'PRED_SLA-{i+1}')

    plt.legend(loc='best', fontsize=7)
    plt.xlabel('Plot Number')
    plt.ylabel('Specific Leaf Area(SLA)')
    plt.title('Calculated Specific Leaf Area of different dates')
    plt.savefig(path, dpi=100)

def save_DAGBScatter(dfd_new, path,n_days, n_plots):
    f= plt.figure(figsize=(12,4))
    markers = ['x', '*', '^', 'o']
    colours = ['b', 'g', 'black', 'red']

    for i in range(n_days):
        plt.scatter(range(len(dfd_new)), dfd_new[f'PRED_AGB-{i+1}'], marker=markers[i%4], color=colours[i%4], label=f'PRED_AGB-{i+1}')

    plt.scatter(range(len(dfd_new)), dfd_new['AGB_maturity'], marker='d', color='orange', label='AGB_maturity')

    plt.legend(loc='best', fontsize=7)
    plt.xlabel('Plot Number')
    plt.ylabel('Above Ground Biomass(AGB)')
    plt.title('Calculated Above Ground Biomass of different dates')
    plt.savefig(path, dpi=100)

def save_DAGBRScatter(dfd_new, path,n_days, n_plots):
    f= plt.figure(figsize=(12,4))
    markers = ['x', '*', '^', 'o']
    colours = ['b', 'g', 'black', 'red']

    for i in range(n_days-1):
        plt.scatter(range(len(dfd_new)), dfd_new[f'AGB_Rate-{i+1}-{i+2}'], marker=markers[i%4], color=colours[i%4], label=f'AGB_Rate-{i+1}-{i+2}')

    plt.scatter(range(len(dfd_new)), dfd_new['MeanAGBRate'], marker=markers[3], color=colours[3], label='MeanAGBRate')

    plt.legend(loc='best', fontsize=7)
    plt.xlabel('Plot Number')
    plt.ylabel('Above Ground Biomass Rate (AGB)')
    plt.title('Above Ground Biomass Rate of different dates')
    plt.savefig(path, dpi=100)
