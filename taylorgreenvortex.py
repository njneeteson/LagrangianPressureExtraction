import numpy as np

def velocity(X,L,V0):
	
	U = np.zeros((X.shape))

	U[:,0] = V0 * np.sin(X[:,0]/L) * np.cos(X[:,1]/L) * np.cos(X[:,2]/L)
	U[:,1] = -V0 * np.cos(X[:,0]/L) * np.sin(X[:,1]/L) * np.cos(X[:,2]/L)
	
	return U

def acceleration(X,L,V0,rho,mu):
	
	DUDt = np.zeros((X.shape))
	
	DUDt[:,0] = (-V0/L) * ( (V0/8)*np.sin(2*X[:,0]/L)*(np.cos(2*X[:,2]/L) + 2) + (2*mu/(rho*L))*np.sin(X[:,0]/L)*np.cos(X[:,1]/L)*np.cos(X[:,2]/L) )
	DUDt[:,1] = (-V0/L) * ( (V0/8)*np.sin(2*X[:,1]/L)*(np.cos(2*X[:,2]/L) + 2) + (2*mu/(rho*L))*np.cos(X[:,0]/L)*np.sin(X[:,1]/L)*np.cos(X[:,2]/L) )
	DUDt[:,2] = (-V0/L) * ( (V0/8)*np.sin(2*X[:,2]/L)*np.cos(2*X[:,0]/L) + (2*mu/(rho*L))*( np.cos(2*X[:,0]/L) + np.cos(2*X[:,1]/L) ) )
	
	return DUDt

def pressure(X,L,V0,rho):
	return (rho*V0**2 / 16) * (np.cos(2*X[:,0]/L) + np.cos(2*X[:,1]/L)) * (np.cos(2*X[:,2]/L) + 2)