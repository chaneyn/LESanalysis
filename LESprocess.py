import datetime
import numpy as np
from numba import jit
import os
import netCDF4 as nc
import scipy.stats
import glob
import pickle
os.system('mkdir -p workspace')
 
def Read_dates():

 #Read in the list of dates
 file = 'LESdates.txt'
 dates = []
 for line in open(file):
     date = datetime.datetime.strptime(line[:-1], "%m-%d-%Y")
     dates.append(date)
 dates = np.array(dates)
 return dates

def Process_LES(dates):
 
 #Collect dates of simulations for both HMG and HTG
 hmg_dates = []
 htg_dates = []
 #HMG
 files = glob.glob('/stor/soteria/hydro/shared/lasso_for_nate/doz_ultlites_01/*')
 for file in files:
     tmp = datetime.datetime.strptime(file.split('/')[-1][12:-3],'%Y-%m-%d_%H:%M')
     hmg_dates.append(tmp)
 hmg_dates = np.sort(np.array(hmg_dates))
 #HTG
 files = glob.glob('/stor/soteria/hydro/shared/lasso_for_nate/doz_ultlites_00/*')
 for file in files:
     tmp = datetime.datetime.strptime(file.split('/')[-1][12:-3],'%Y-%m-%d_%H:%M')
     htg_dates.append(tmp)
 htg_dates = np.sort(np.array(htg_dates))
 
 #Read in the HMG/HTG TKE + LWP for ones that are finished
 odb = {}
 count = 0
 for date in dates:
     count += 1
     print(count,date)
     tdate = date + datetime.timedelta(hours=12)
     mdates = []
     for i in range(15*6+1):
         mdates.append(tdate+i*datetime.timedelta(minutes=10))
     mdates = np.array(mdates)
     #Create list of dates
     strdate = date.strftime('%Y-%m-%d')
     odb[strdate] = {'HMG':{'LWP':np.array([]),'TKE':np.array([])},'HTG':{'LWP':np.array([]),'TKE':np.array([])}}
     #HMG
     odb[strdate]['HMG']['TKE'] = -9999*np.ones(mdates.size)
     odb[strdate]['HMG']['LWP'] = -9999*np.ones(mdates.size)
     m = (hmg_dates >= tdate) & (hmg_dates <= tdate+datetime.timedelta(hours=16))
     for tmp in hmg_dates[m]:
         file = '/stor/soteria/hydro/shared/lasso_for_nate/doz_ultlites_01/ultlite_d01_%d-%02d-%02d_%02d:%02d:00' % (tmp.year,tmp.month,tmp.day,tmp.hour,tmp.minute)
         fp = nc.Dataset(file)
         #Create list of dates inside
         tmdates = []
         for i in range(fp['CST_TKE'][:].size):
             tmdates.append(tmp + i*datetime.timedelta(minutes=10))
         tmdates = np.array(tmdates)
         CST_TKE = fp['CST_TKE'][:]
         CST_LWP = fp['CST_LWP'][:]
         for itm in range(tmdates.size):
             odb[strdate]['HMG']['TKE'][mdates == tmdates[itm]] = CST_TKE[itm] 
             odb[strdate]['HMG']['LWP'][mdates == tmdates[itm]] = CST_LWP[itm] 
     #break
     #HTG
     odb[strdate]['HTG']['TKE'] = -9999*np.ones(mdates.size)
     odb[strdate]['HTG']['LWP'] = -9999*np.ones(mdates.size)
     m = (htg_dates >= tdate) & (htg_dates <= tdate+datetime.timedelta(hours=17))
     for tmp in htg_dates[m]:
         file = '/stor/soteria/hydro/shared/lasso_for_nate/doz_ultlites_00/ultlite_d01_%d-%02d-%02d_%02d:%02d:00' % (tmp.year,tmp.month,tmp.day,tmp.hour,tmp.minute)
         fp = nc.Dataset(file)
         #Create list of dates inside
         tmdates = []
         for i in range(fp['CST_TKE'][:].size):
             tmdates.append(tmp + i*datetime.timedelta(minutes=10))
         tmdates = np.array(tmdates)
         CST_TKE = fp['CST_TKE'][:]
         CST_LWP = fp['CST_LWP'][:]
         for itm in range(tmdates.size):
             odb[strdate]['HTG']['TKE'][mdates == tmdates[itm]] = CST_TKE[itm] 
             odb[strdate]['HTG']['LWP'][mdates == tmdates[itm]] = CST_LWP[itm]
     #Clean up 
     for var in ['TKE','LWP']:
         m = (odb[strdate]['HTG'][var] == -9999) | (odb[strdate]['HMG'][var] == -9999)
         odb[strdate]['HTG'][var][m] = -9999
         odb[strdate]['HMG'][var][m] = -9999

 #Save the data
 file = 'workspace/LESoutput.pck'
 pickle.dump(odb,open(file,'wb'))
 return

def Process_surface_fields(dates):

 db = {}
 for date in dates:
    print(date)
    db[date] = {}
    data = []
    for hour in range(15,22):
     file = '/stor/soteria/hydro/shared/lasso_for_nate/dx0250nx0401/jsssh_bdy_02_%d-%02d-%02d-%2d-00' % (date.year,date.month,date.day,hour)
     #file = '/stor/soteria/hydro/shared/lasso_for_nate/dx0250nx0401/jsslw_bdy_02_%d-%02d-%02d-%2d-00' % (date.year,date.month,date.day,hour)
     fp = open(file)
     data.append(np.reshape(np.loadtxt(file),(401,401)))
    data = np.array(data)
    data = np.mean(data,axis=0)
    db[date]['SH'] = data
 #Save the data
 file = 'workspace/SurfaceFields.pck'
 pickle.dump(db,open(file,'wb'))
 return

@jit(nopython=True)
def calculate_ns(data,h,dx,mu,step):
    Q = 0.0
    count = 0
    inew = int(h/dx)
    for i in range(0,data.shape[0],step):
        for j in range(0,data.shape[1],step):
            iinew = i + inew
            if iinew >= data.shape[0]:iinew = iinew % data.shape[0]
            Q += (data[i,j]-mu)*(data[iinew,j]-mu)
            count += 1
            iinew = i - inew
            if iinew < 0:iinew = iinew % data.shape[0]
            Q += (data[i,j]-mu)*(data[iinew,j]-mu)
            count += 1
    Q = Q/count

    return Q

@jit(nopython=True)
def calculate_ew(data,h,dx,mu,step):
    Q = 0.0
    count = 0
    jnew = int(h/dx)
    for i in range(0,data.shape[0],step):
        for j in range(0,data.shape[1],step):
            jjnew = j + jnew
            if jjnew >= data.shape[1]:jjnew = jjnew % data.shape[1]
            Q += (data[i,j]-mu)*(data[i,jjnew]-mu)
            count += 1
            jjnew = j - jnew
            if jjnew < 0:jjnew = jjnew % data.shape[1]
            Q += (data[i,j]-mu)*(data[i,jjnew]-mu)
            count += 1
    Q = Q/count

    return Q

@jit(nopython=True)
def calculate_swne(data,h,dx,mu,step):
    Q = 0.0
    count = 0
    jnew = int(h/dx)
    inew = int(h/dx)
    for i in range(0,data.shape[0],step):
        for j in range(0,data.shape[1],step):
            iinew = i + inew
            jjnew = j + jnew
            if iinew >= data.shape[1]:iinew = iinew % data.shape[0]
            if jjnew >= data.shape[1]:jjnew = jjnew % data.shape[1]
            Q += (data[i,j]-mu)*(data[iinew,jjnew]-mu)
            count += 1
            iinew = i - inew
            jjnew = j - jnew
            if iinew < 0:iinew = iinew % data.shape[0]
            if jjnew < 0:jjnew = jjnew % data.shape[1]
            Q += (data[i,j]-mu)*(data[iinew,jjnew]-mu)
            count += 1
    Q = Q/count

    return Q

@jit(nopython=True)
def calculate_nwse(data,h,dx,mu,step):
    Q = 0.0
    count = 0
    jnew = int(h/dx)
    inew = int(h/dx)
    for i in range(0,data.shape[0],step):
        for j in range(0,data.shape[1],step):
            iinew = i - inew
            jjnew = j + jnew
            if iinew < 0:iinew = iinew % data.shape[0]
            if jjnew >= data.shape[1]:jjnew = jjnew % data.shape[1]
            Q += (data[i,j]-mu)*(data[iinew,jjnew]-mu)
            count += 1
            iinew = i + inew
            jjnew = j - jnew
            if iinew >= data.shape[1]:iinew = iinew % data.shape[0]
            if jjnew < 0:jjnew = jjnew % data.shape[1]
            Q += (data[i,j]-mu)*(data[iinew,jjnew]-mu)
            count += 1
    Q = Q/count

    return Q
    
def calculate_anisotropic_covariance_functions(data,hs,dx=250):
    step = 1
    #calculate mean
    mu = np.mean(data)
    Qns = np.zeros(len(hs))
    for ih in range(len(hs)):
        Qns[ih] = calculate_ns(data,hs[ih],dx,mu,step)
    Qew = np.zeros(len(hs))
    for ih in range(len(hs)):
        Qew[ih] = calculate_ew(data,hs[ih],dx,mu,step)
    Qswne = np.zeros(len(hs))
    for ih in range(len(hs)):
        Qswne[ih] = calculate_swne(data,hs[ih],dx,mu,step)
    Qnwse = np.zeros(len(hs))
    for ih in range(len(hs)):
        Qnwse[ih] = calculate_nwse(data,hs[ih],dx,mu,step)
    #Interpolate qswne and qnwse to be on same h as ns/ew
    hsdiag = np.sqrt(2)*hs
    #interpolate
    Qswne= np.interp(hs,hsdiag,Qswne)
    Qnwse= np.interp(hs,hsdiag,Qnwse)
    return (Qns,Qew,Qswne,Qnwse)

def Compute_Surface_Fields_Stats(dates):

 #Read in sounding info
 file = 'workspace/LASSOsoundings.pck'
 odb = pickle.load(open(file,'rb'))

 hs = np.array([0,250,1000,2000,3000,4000,5000,7500,10000,12500,15000,17500,20000,25000,30000,35000,40000,45000,50000])
 odb['SH_Qew'] = []
 odb['SH_Qns'] = []
 odb['SH_Qswne'] = []
 odb['SH_Qnwse'] = []
 odb['SH_var'] = []
 odb['SH_mean'] = []
 odb['SH_data'] = []
 odb['SH_skew'] = []
 odb['SH_kurt'] = []
 db = pickle.load(open('workspace/SurfaceFields.pck','rb'))
 for date in dates:
  data = db[date]['SH']
  #calculate anisotropic covariance functions
  (Qns,Qew,Qswne,Qnwse) = calculate_anisotropic_covariance_functions(data,hs,dx=250)
  #save info
  odb['SH_Qns'].append(Qns)
  odb['SH_Qew'].append(Qew)
  odb['SH_Qswne'].append(Qswne)
  odb['SH_Qnwse'].append(Qnwse)
  odb['SH_var'].append(np.var(data))
  odb['SH_mean'].append(np.mean(data))
  odb['SH_skew'].append(scipy.stats.skew(data.flatten()))
  odb['SH_kurt'].append(scipy.stats.kurtosis(data.flatten()))
 for var in odb:
  odb[var] = np.array(odb[var])

 #Compute covariance functions parallel and perpendicular to the geostrophic wind
 #Parallel
 Q0 = []
 for i in range(odb['theta'].size):
     theta = np.degrees(odb['theta'][i])
     if ((theta > 0) & (theta <= 45)):
         dt = np.radians(theta - 0)
         Q0.append(dt/(np.pi/4)*odb['SH_Qswne'][i,:] + (1-dt/(np.pi/4))*odb['SH_Qew'][i,:])
     elif ((theta > 45) & (theta <= 90)):
         dt = np.radians(theta - 45)
         Q0.append(dt/(np.pi/4)*odb['SH_Qns'][i,:] + (1-dt/(np.pi/4))*odb['SH_Qswne'][i,:])
     elif (theta > 90) & (theta <= 135):
         dt = np.radians(theta - 90)
         Q0.append(dt/(np.pi/4)*odb['SH_Qnwse'][i,:] + (1-dt/(np.pi/4))*odb['SH_Qns'][i,:])
     elif (theta > 135) & (theta <= 180):
         dt = np.radians(theta - 135)
         Q0.append(dt/(np.pi/4)*odb['SH_Qew'][i,:] + (1-dt/(np.pi/4))*odb['SH_Qnwse'][i,:])
 Q0 = np.array(Q0)
 odb['SH_Q0'] = np.copy(Q0)
 #Perpendicular
 Q90 = []
 for i in range(odb['theta'].size):
     theta = 90+np.degrees(odb['theta'][i])
     if theta > 180:
         theta = theta - 180
     if ((theta > 0) & (theta <= 45)):
         dt = np.radians(theta - 0)
         Q90.append(dt/(np.pi/4)*odb['SH_Qswne'][i] + (1-dt/(np.pi/4))*odb['SH_Qew'][i])
     elif ((theta > 45) & (theta <= 90)):
         dt = np.radians(theta - 45)
         Q90.append(dt/(np.pi/4)*odb['SH_Qns'][i] + (1-dt/(np.pi/4))*odb['SH_Qswne'][i])
     elif (theta > 90) & (theta <= 135):
         dt = np.radians(theta - 90)
         Q90.append(dt/(np.pi/4)*odb['SH_Qnwse'][i] + (1-dt/(np.pi/4))*odb['SH_Qns'][i])
     elif (theta > 135) & (theta <= 180):
         dt = np.radians(theta - 135)
         Q90.append(dt/(np.pi/4)*odb['SH_Qew'][i] + (1-dt/(np.pi/4))*odb['SH_Qnwse'][i])
 Q90 = np.array(Q90)
 odb['SH_Q90'] = np.copy(Q90)

 #Calculate correlation lengths
 for thld in [0.25,0.50,0.75]:
  L0 = []
  L90 = []
  for i in range(odb['SH_Q0'].shape[0]):
      hsnew = np.linspace(0,50000,2500)
      #parallel to flow
      sh_q0_new = np.interp(hsnew,hs,odb['SH_Q0'][i,:])
      ins = sh_q0_new <= thld*sh_q0_new[0]
      if np.sum(ins) == 0:L0.append(50.0) #km
      else: L0.append(hsnew[ins][0]/1000.0) #km
      #perpendicular to flow
      sh_q90_new = np.interp(hsnew,hs,odb['SH_Q90'][i,:])
      ins = sh_q90_new <= thld*sh_q90_new[0]
      if np.sum(ins) == 0:L90.append(50.0) #km
      else: L90.append(hsnew[ins][0]/1000.0) #km
  L0 = np.array(L0)
  odb['SH_L0_%.2f' % thld] = np.copy(L0)
  L90 = np.array(L90)
  odb['SH_L90_%.2f' % thld] = np.copy(L90)
 
 #Save the data
 file = 'workspace/SFstats.pck'
 pickle.dump(odb,open(file,'wb'))

 return

def Process_sounding_data(dates):

 db = {}
 db['ws'] = []
 db['theta'] = []
 for date in dates:
    print(date)
    file = '/stor/soteria/hydro/shared/lasso_for_nate/lasso_download/sgp%d%02d%02d/config/input_sounding' % (date.year,date.month,date.day)
    data = np.loadtxt(file,skiprows=1)
    z = data[:,0]
    dz = z[1:]-z[:-1]
    iz = (z[1:] >= 2000) & (z[1:] <= 20000)#20000)
    zav = (z[1:] + z[1:])/2
    f = dz[iz]/np.sum(dz[iz])
    u = ((data[1:,-2] + data[0:-1,-2])/2)[iz]
    v = ((data[1:,-1] + data[0:-1,-1])/2)[iz]
    ug = np.sum(f*u)
    vg = np.abs(np.sum(f*v))
    #Calculate magnitude
    db['ws'].append((ug**2 + vg**2)**0.5)
    db['theta'].append(np.arctan2(vg,ug)) #only 0-180 degrees (on purpose)
    print(date,np.arctan2(vg,ug),np.degrees(np.arctan2(vg,ug)))
 for var in db:
    db[var] = np.array(db[var])
 #Save the data
 file = 'workspace/LASSOsoundings.pck'
 pickle.dump(db,open(file,'wb'))
 
 return
 
#Read dates
dates = Read_dates()

#Process LES output
Process_LES(dates)

#Process surface fields
Process_surface_fields(dates)

#Process sounding data
Process_sounding_data(dates)

#Compute surface field stats
Compute_Surface_Fields_Stats(dates)
