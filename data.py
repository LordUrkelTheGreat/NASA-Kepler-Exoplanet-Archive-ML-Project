# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


"""
This function will read the Kepler Exoplanet Dataset, preprocess the data, and split the
data into features and targets
"""
def extract_data():
    # load in the kepler exoplanet dataset
    #kepler_exoplanet_dataset = pd.read_csv('cumulative_2024.02.18_16.53.59.csv', on_bad_lines = 'skip')
    kepler_exoplanet_dataset = pd.read_csv('Kepler_Exoplanet_Archive.csv')

    # copy the dataset just in case we mess with any data
    copy_kepler = kepler_exoplanet_dataset.copy(deep = True)

    # rename the columns in the dataset
    copy_kepler = copy_kepler.rename(columns =
        {
            'rowid'                 : 'RowID',
            'kepid'                 : 'KepID',
            'kepoi_name'            : 'KOI Name',
            'kepler_name'           : 'Kepler Name',
            'koi_disposition'       : 'Exoplanet Archive Disposition',
            'koi_vet_stat'          : 'Vetting Status',
            'koi_vet_date'          : 'Date of Last Parameter Update',
            'koi_pdisposition'      : 'Disposition Using Kepler Data',
            'koi_score'             : 'Disposition Score',
            'koi_fpflag_nt'         : 'Not Transit-Like False Positive Flag',
            'koi_fpflag_ss'         : 'Stellar Eclipse False Positive Flag',
            'koi_fpflag_co'         : 'Centroid Offset False Positive Flag',
            'koi_fpflag_ec'         : 'Ephemeris Match Indicates Contamination False Positive Flag',
            'koi_disp_prov'         : 'Disposition Provenance',
            'koi_comment'           : 'Comment',
            'koi_period'            : 'Orbital Period [days]',
            'koi_period_err1'       : 'Orbital Period Upper Unc. [days]',
            'koi_period_err2'       : 'Orbital Period Lower Unc. [days]',
            'koi_time0bk'           : 'Transit Epoch [BKJD]',
            'koi_time0bk_err1'      : 'Transit Epoch Upper Unc. [BKJD]',
            'koi_time0bk_err2'      : 'Transit Epoch Lower Unc. [BKJD]',
            'koi_time0'             : 'Transit Epoch [BJD]',
            'koi_time0_err1'        : 'Transit Epoch Upper Unc. [BJD]',
            'koi_time0_err2'        : 'Transit Epoch Lower Unc. [BJD]',
            'koi_eccen'             : 'Eccentricity',
            'koi_eccen_err1'        : 'Eccentricity Upper Unc.',
            'koi_eccen_err2'        : 'Eccentricity Lower Unc.',
            'koi_longp'             : 'Long. of Periastron [deg]',
            'koi_longp_err1'        : 'Long. of Periastron Upper Unc. [deg]',
            'koi_longp_err2'        : 'Long. of Periastron Lower Unc. [deg]',
            'koi_impact'            : 'Impact Parameter',
            'koi_impact_err1'       : 'Impact Parameter Upper Unc.',
            'koi_impact_err2'       : 'Impact Parameter Lower Unc.',
            'koi_duration'          : 'Transit Duration [hrs]',
            'koi_duration_err1'     : 'Transit Duration Upper Unc. [hrs]',
            'koi_duration_err2'     : 'Transit Duration Lower Unc. [hrs]',
            'koi_ingress'           : 'Ingress Duration [hrs]',
            'koi_ingress_err1'      : 'Ingress Duration Upper Unc. [hrs]',
            'koi_ingress_err2'      : 'Ingress Duration Lower Unc. [hrs]',
            'koi_depth'             : 'Transit Depth [ppm]',
            'koi_depth_err1'        : 'Transit Depth Upper Unc. [ppm]',
            'koi_depth_err2'        : 'Transit Depth Lower Unc. [ppm]',
            'koi_ror'               : 'Planet-Star Radius Ratio',
            'koi_ror_err1'          : 'Planet-Star Radius Ratio Upper Unc.',
            'koi_ror_err2'          : 'Planet-Star Radius Ratio Lower Unc.',
            'koi_srho'              : 'Fitted Stellar Density [g/cm**3]',
            'koi_srho_err1'         : 'Fitted Stellar Density Upper Unc. [g/cm**3]',
            'koi_srho_err2'         : 'Fitted Stellar Density Lower Unc. [g/cm**3]',
            'koi_fittype'           : 'Planetary Fit Type',
            'koi_prad'              : 'Planetary Radius [Earth radii]',
            'koi_prad_err1'         : 'Planetary Radius Upper Unc. [Earth radii]',
            'koi_prad_err2'         : 'Planetary Radius Lower Unc. [Earth radii]',
            'koi_sma'               : 'Orbit Semi-Major Axis [au]',
            'koi_sma_err1'          : 'Orbit Semi-Major Axis Upper Unc. [au]',
            'koi_sma_err2'          : 'Orbit Semi-Major Axis Lower Unc. [au]',
            'koi_incl'              : 'Inclination [deg]',
            'koi_incl_err1'         : 'Inclination Upper Unc. [deg]',
            'koi_incl_err2'         : 'Inclination Lower Unc. [deg]',
            'koi_teq'               : 'Equilibrium Temperature [K]',
            'koi_teq_err1'          : 'Equilibrium Temperature Upper Unc. [K]',
            'koi_teq_err2'          : 'Equilibrium Temperature Lower Unc. [K]',
            'koi_insol'             : 'Insolation Flux [Earth flux]',
            'koi_insol_err1'        : 'Insolation Flux Upper Unc. [Earth flux]',
            'koi_insol_err2'        : 'Insolation Flux Lower Unc. [Earth flux]',
            'koi_dor'               : 'Planet-Star Distance over Star Radius',
            'koi_dor_err1'          : 'Planet-Star Distance over Star Radius Upper Unc.',
            'koi_dor_err2'          : 'Planet-Star Distance over Star Radius Lower Unc.',
            'koi_limbdark_mod'      : 'Limb Darkening Model',
            'koi_ldm_coeff4'        : 'Limb Darkening Coeff. 4',
            'koi_ldm_coeff3'        : 'Limb Darkening Coeff. 3',
            'koi_ldm_coeff2'        : 'Limb Darkening Coeff. 2',
            'koi_ldm_coeff1'        : 'Limb Darkening Coeff. 1',
            'koi_parm_prov'         : 'Parameters Provenance',
            'koi_max_sngle_ev'      : 'Maximum Single Event Statistic',
            'koi_max_mult_ev'       : 'Maximum Multiple Event Statistic',
            'koi_model_snr'         : 'Transit Signal-to-Noise',
            'koi_count'             : 'Number of Planets',
            'koi_num_transits'      : 'Number of Transits',
            'koi_tce_plnt_num'      : 'TCE Planet Number',
            'koi_tce_delivname'     : 'TCE Delivery',
            'koi_quarters'          : 'Quarters',
            'koi_bin_oedp_sig'      : 'Odd-Even Depth Comparision Statistic',
            'koi_trans_mod'         : 'Transit Model',
            'koi_model_dof'         : 'Degrees of Freedom',
            'koi_model_chisq'       : 'Chi-Square',
            'koi_datalink_dvr'      : 'Link to DV Report',
            'koi_datalink_dvs'      : 'Link to DV Summary',
            'koi_steff'             : 'Stellar Effective Temperature [K]',
            'koi_steff_err1'        : 'Stellar Effective Temperature Upper Unc. [K]',
            'koi_steff_err2'        : 'Stellar Effective Temperature Lower Unc. [K]',
            'koi_slogg'             : 'Stellar Surface Gravity [log10(cm/s**2)]',
            'koi_slogg_err1'        : 'Stellar Surface Gravity Upper Unc. [log10(cm/s**2)]',
            'koi_slogg_err2'        : 'Stellar Surface Gravity Lower Unc. [log10(cm/s**2)]',
            'koi_smet'              : 'Stellar Metallicity [dex]',
            'koi_smet_err1'         : 'Stellar Metallicity Upper Unc. [dex]',
            'koi_smet_err2'         : 'Stellar Metallicity Lower Unc. [dex]',
            'koi_srad'              : 'Stellar Radius [Solar radii]',
            'koi_srad_err1'         : 'Stellar Radius Upper Unc. [Solar radii]',
            'koi_srad_err2'         : 'Stellar Radius Lower Unc. [Solar radii]',
            'koi_smass'             : 'Stellar Mass [Solar mass]',
            'koi_smass_err1'        : 'Stellar Mass Upper Unc. [Solar mass]',
            'koi_smass_err2'        : 'Stellar Mass Lower Unc. [Solar mass]',
            'koi_sage'              : 'Stellar Age [Gyr]',
            'koi_sage_err1'         : 'Stellar Age Upper Unc. [Gyr]',
            'koi_sage_err2'         : 'Stellar Age Lower Unc. [Gyr]',
            'koi_sparprov'          : 'Stellar Parameter Provenance',
            'ra'                    : 'RA [decimal degrees]',
            'dec'                   : 'Dec [decimal degrees]',
            'koi_kepmag'            : 'Kepler-band [mag]',
            'koi_gmag'              : 'g-band [mag]',
            'koi_rmag'              : 'r-band [mag]',
            'koi_imag'              : 'i-band [mag]',
            'koi_zmag'              : 'z-band [mag]',
            'koi_jmag'              : 'J-band [mag]',
            'koi_hmag'              : 'H-band [mag]',
            'koi_kmag'              : 'K-band [mag]',
            'koi_fwm_stat_sig'      : 'FW Offset Significance [percent]',
            'koi_fwm_sra'           : 'FW Source &alpha;(OOT) [hrs]',
            'koi_fwm_sra_err'       : 'FW Source &alpha;(OOT) Unc. [hrs]',
            'koi_fwm_sdec'          : 'FW Source &delta;(OOT) [deg]',
            'koi_fwm_sdec_err'      : 'FW Source &delta;(OOT) Unc. [deg]',
            'koi_fwm_srao'          : 'FW Source &Delta;&alpha;(OOT) [sec]',
            'koi_fwm_srao_err'      : 'FW Source &Delta;&alpha;(OOT) Unc. [sec]',
            'koi_fwm_sdeco'         : 'FW Source &Delta;&delta;(OOT) [arcsec]',
            'koi_fwm_sdeco_err'     : 'FW Source &Delta;&delta;(OOT) Unc. [arcsec]',
            'koi_fwm_prao'          : 'FW &Delta;&alpha;(OOT) [sec]',
            'koi_fwm_prao_err'      : 'FW &Delta;&alpha;(OOT) Unc. [sec]',
            'koi_fwm_pdeco'         : 'FW &Delta;&delta;(OOT) [arcsec]',
            'koi_fwm_pdeco_err'     : 'FW &Delta;&delta;(OOT) Unc. [arcsec]',
            'koi_dicco_mra'         : 'PRF &Delta;&alpha;<sub>SQ</sub>(OOT) [arcsec]',
            'koi_dicco_mra_err'     : 'PRF &Delta;&alpha;<sub>SQ</sub>(OOT) Unc. [arcsec]',
            'koi_dicco_mdec'        : 'PRF &Delta;&delta;<sub>SQ</sub>(OOT) [arcsec]',
            'koi_dicco_mdec_err'    : 'PRF &Delta;&delta;<sub>SQ</sub>(OOT) Unc. [arcsec]',
            'koi_dicco_msky'        : 'PRF &Delta;&theta;<sub>SQ</sub>(OOT) []arcsec',
            'koi_dicco_msky_err'    : 'PRF &Delta;&theta;<sub>SQ</sub>(OOT) Unc. [arcsec]',
            'koi_dikco_mra'         : 'PRF &Delta;&alpha;<sub>SQ</sub>(KIC) [arcsec]',
            'koi_dikco_mra_err'     : 'PRF &Delta;&alpha;<sub>SQ</sub>(KIC) Unc. [arcsec]',
            'koi_dikco_mdec'        : 'PRF &Delta;&delta;<sub>SQ</sub>(KIC) [arcsec]',
            'koi_dikco_mdec_err'    : 'PRF &Delta;&delta;<sub>SQ</sub>(KIC) Unc. [arcsec]',
            'koi_dikco_msky'        : 'PRF &Delta;&theta;<sub>SQ</sub>(KIC) [arcsec]',
            'koi_dikco_msky_err'    : 'PRF &Delta;&theta;<sub>SQ</sub>(KIC) Unc. [arcsec]',
        })

    # print the first 5 rows of the dataset
    #print("Kepler Exoplanet Dataset Head:\n", copy_kepler.head(), "\n")

    # check which columns have null values
    #print("Number of NaN values per column:\n", copy_kepler.isnull().sum(), "\n")

    # show the names of all columns
    #print(list(copy_kepler.columns))

    """
    NOTE: These rows need to be extracted as they are either IDs, comments, don't have any data, or is data
        that can't be converted into a numerical value

    RowID, KepID, KOI Name, Kepler Name, Vetting Status, Date of Last Parameter Update, Disposition Provenance,
    Comment, Eccentricity Upper Unc. [BJD], Eccentricity Lower Unc. [BJD], Long. of Periastron [deg],
    Long. of Periastron Upper Unc. [deg], Long. of Periastron Lower Unc. [deg], Ingress Duration [hrs],
    Ingress Duration Upper Unc. [hrs], Ingress Duration Lower Unc. [hrs], Planetary Fit Type,
    Orbit Semi-Major Axis Upper Unc. [au], Orbit Semi-Major Axis Lower Unc. [au], Inclination Upper Unc. [deg],
    Inclination Lower Unc. [deg], Equilibrium Temperature Upper Unc. [K], Equilibrium Temperature Lower Unc. [K],
    Limb Darkening Model, Parameters Provenance, TCE Planet Number, TCE Delivery, Transit Model, Degrees of Freedom,
    Chi-Square, Link to DV Report, Link to DV Summary, Stellar Age [Gyr], Stellar Age Upper Unc. [Gyr], Stellar Age Lower Unc. [Gyr]
    """
    copy_kepler.drop(columns = ['RowID', 'KepID', 'KOI Name', 'Kepler Name', 'Vetting Status', 'Date of Last Parameter Update', 'Disposition Provenance', 'Comment', 'Eccentricity Upper Unc.', 'Eccentricity Lower Unc.', 'Long. of Periastron [deg]', 'Long. of Periastron Upper Unc. [deg]', 'Long. of Periastron Lower Unc. [deg]', 'Ingress Duration [hrs]', 'Ingress Duration Upper Unc. [hrs]', 'Ingress Duration Lower Unc. [hrs]', 'Planetary Fit Type', 'Orbit Semi-Major Axis Upper Unc. [au]', 'Orbit Semi-Major Axis Lower Unc. [au]', 'Inclination Upper Unc. [deg]', 'Inclination Lower Unc. [deg]', 'Equilibrium Temperature Upper Unc. [K]', 'Equilibrium Temperature Lower Unc. [K]', 'Limb Darkening Model', 'TCE Planet Number', 'TCE Delivery', 'Transit Model', 'Degrees of Freedom', 'Chi-Square', 'Link to DV Report', 'Link to DV Summary', 'Stellar Age [Gyr]', 'Stellar Age Upper Unc. [Gyr]', 'Stellar Age Lower Unc. [Gyr]'], inplace = True)

    # print the first 5 rows of the dataset
    #print("Kepler Exoplanet Dataset Head:\n", copy_kepler.head(), "\n")

    # check which columns have null values
    #print("Number of NaN values per column:\n", copy_kepler.isnull().sum(), "\n")

    # show the names of all columns
    print(list(copy_kepler.columns), "\n")

    for column in copy_kepler.columns:
        print(column, ": ", pd.unique(copy_kepler[column]), "\n")

    # see the different values in this column
    #print(pd.unique(copy_kepler['Exoplanet Archive Disposition']), "\n")

    # set strings to numerical values
    copy_kepler['Exoplanet Archive Disposition'] = copy_kepler['Exoplanet Archive Disposition'].apply(lambda i: 0 if i == 'CONFIRMED' else 1 if i == 'CANDIDATE' else 2)

    # see the different values in this column
    #print(pd.unique(copy_kepler['Exoplanet Archive Disposition']), "\n")

    # see the different values in this column
    #print(pd.unique(copy_kepler['Disposition Using Kepler Data']), "\n")

    # set strings to numerical values
    copy_kepler['Disposition Using Kepler Data'] = copy_kepler['Disposition Using Kepler Data'].apply(lambda i: 0 if i == 'CANDIDATE' else 1)

    # see the different values in this column
    #print(pd.unique(copy_kepler['Disposition Using Kepler Data']), "\n")

    # see the different values in this column
    #print(pd.unique(copy_kepler['Parameters Provenance']), "\n")

    # set strings to numerical values
    copy_kepler['Parameters Provenance'] = copy_kepler['Parameters Provenance'].apply(lambda i: 0 if i == 'q1_q17_dr25_koi' else 1 if i == 'q1_q16_koi' else 2)

    # see the different values in this column
    #print(pd.unique(copy_kepler['Parameters Provenance']), "\n")

    return

extract_data()