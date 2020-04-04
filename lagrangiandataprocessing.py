import numpy as np
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from scipy import sparse
from scipy.sparse.linalg import gmres

# first we need a "small-scale" tessellation function
def tessellateSS(Xc,Xn):
	# Xc - central point, shape 1 x nDim
	# Xn - neighbour points, shape nNeighbours x nDim
	
	numNeighbours = Xn.shape[0]
	
	# first calculate the physical distance between the central point and each neighbour
	H = np.linalg.norm(np.subtract(Xc,Xn),axis=1).reshape(numNeighbours,1)
	
	# then calculate the normal vector pointing from the central point to each neighbour
	nHat = np.divide(np.subtract(Xn,Xc),H)
	
	# now the hard part: calculate the size of the voronoi cell face corresponding to each neighbour
	
	# compute the voronoi diagram for all of the points together
	vor = Voronoi(np.concatenate(([Xc],Xn)))
	
	# then compute the convex hull over the voronoi vortices of the region corresponding to the center point
	hull = ConvexHull(vor.vertices[vor.regions[vor.point_region[0]]])
	
	# using the convex hull, we can now identify the voronoi faces of the center point
	# this array has the shape nFaces by nDim, it is a list of faces where each face is a list of
	# indices into the voronoi vertices
	vorFaces = np.asarray(vor.regions[vor.point_region[0]])[
		hull.simplices.reshape(hull.simplices.size)].reshape(hull.simplices.shape)
	
	numFaces = vorFaces.shape[0]
	
	# initialize the voronoi cell face size vector as 0 for each neighbour
	S = np.zeros((numNeighbours,1))
	
	# now we're going to loop through all of the faces and neighbours and match them up
	for f in range(numFaces):
		
		# first we will calculate the area of the face
		faceArea = triangleArea3D(vor.vertices[vorFaces[f]])
		
		# then loop over the neighbours
		for n in range(numNeighbours):
			
			# in order to check if a face corresponds to a neighbour, the criteria is that
			# all of the vertices of face 'f' ( vorFaces[f] ) are contained in the list of
			# voronoi vertices for neighbour 'n' ( vor.regions[n+1] )
			if all(v in vor.regions[vor.point_region[n+1]] for v in vorFaces[f]):
				S[n] = S[n] + faceArea
				break
	
	return H, S, nHat


# this is the main tessellation function that will construct the network for the full dataset
def tessellate(X):
	
	########## INITIALIZATION
	
	# X should be of shape num points by dim
	numPoints = X.shape[0]
	
	# use built-in functions to get the triangulation and tessellation quickly
	tri = Delaunay(X)
	vor = Voronoi(X)
	
	# now we're going to use 'tri' and 'vor' to construct a network on the points
	# that can be used to perform vector calculus operations
	
	# first we create a vector B of the size num points by 1
	B = np.zeros((numPoints,1), dtype=bool)
	
	# B tells us whether or not the point with the same index is bounded,
	# bounded meaning it has a closed voronoi region
	# a quick way to check if B is bounded is to go through all of the voronoi regions,
	# of which there is 1 per point in X, and each of which contains a list of 
	# voronoi indices defining the region. an index of -1 means that one of the indices
	# is outside of the voronoi diagram and thus the region, and point, is unbounded
	# if there is no -1 in the list, the region is bounded
	for p in range(numPoints):
		B[p] = vor.regions[vor.point_region[p]].count(-1) == 0
	
	# Now let's initialize the other things this function is going to output in
	# the final network dictionary
	
	N =	   [[] for i in range(numPoints)] # index of neighbours
	H =	   [[] for i in range(numPoints)] # distance to neighbours
	S =	   [[] for i in range(numPoints)] # size of cell faces
	nHat = [[] for i in range(numPoints)] # normal vector of faces
	
	
	########## NEIGHBOUR DETERMINATION
	
	# we're going to determine which points are connected to each other by running
	# through the triangulation and saying that any two points that are in the same
	# simplex are connected
	
	for t in range(len(tri.simplices)):
		
		for i in range(len(tri.simplices[t])-1):
			for j in range(len(tri.simplices[t])-i):
				
				p = tri.simplices[t,i]
				q = tri.simplices[t,i+j]
				
				if B[p] or B[q]:
					
					N[p].append(q)
					N[q].append(p)

	# then we clean up the neighbours lists to be unique and not contain the center point
	
	for p in range(numPoints):
		N[p] = list(np.unique(N[p]))
		if p in N[p]: N[p].remove(p)
	
	########## PARAMETER COMPUTATION
	
	# first iterate over all the points and compute parameters for bounded points
	
	for p in range(numPoints):
		
		if B[p]:
			
			H[p], S[p], nHat[p] = tessellateSS(X[p],X[N[p]])
	
	# now iterate over all of the unbounded points and assign their
	# parameters by looking at their neighbours and giving them the corresponding
	# parameter values: any connection in the network has only one length/size/normVec
	
	for p in range(numPoints):
		
		if not(B[p]):
			
			for n in N[p]:
				
				H[p].append(H[n][N[n].index(p)])
				S[p].append(S[n][N[n].index(p)])
				nHat[p].append(nHat[n][N[n].index(p)])
			
			H[p] = np.asarray(H[p])
			S[p] = np.asarray(S[p])
			nHat[p] = np.asarray(nHat[p])
			
	########## ORGANIZE OUTPUTS
	
	# output everything in a single dictionary
	
	return {'N':N, 'B':B, 'H':H, 'S':S, 'nHat':nHat}

# this function is useful for calculating face sizes cleanly
def triangleArea3D(V):
	return (1/2)*np.linalg.norm(np.cross( V[1]-V[0], V[2]-V[0] ))
	

def divergence(network,F):
	
	divF = np.zeros((F.shape[0],1))
	
	for p in range(F.shape[0]):
		
		Np = np.asarray(network['N'][p]).reshape(len(network['N'][p]),1)
		Hp = np.asarray(network['H'][p]).reshape(len(network['H'][p]),1)
		Sp = np.asarray(network['S'][p]).reshape(len(network['S'][p]),1)
		nHatp = np.asarray(network['nHat'][p])
		
		Fp = np.asarray(F[p,:]).reshape(1,len(F[p,:]))
		Fn = np.asarray(F[Np,:]).reshape(len(network['N'][p]),len(F[p,:]))
		
		divF[p] = np.divide(
			np.sum(np.matmul((Fn + Fp*np.ones(Np.shape)).T,Sp)), 
			np.sum(np.multiply(Sp,Hp))/3 
			)

	return divF


def laplacian(network,f):
	
	lapf = np.zeros(f.shape)
	
	for p in range(f.shape[0]):
		
		Np = np.asarray(network['N'][p]).reshape(len(network['N'][p]),1)
		Hp = np.asarray(network['H'][p]).reshape(len(network['H'][p]),1)
		Sp = np.asarray(network['S'][p]).reshape(len(network['S'][p]),1)
		
		lapf[p] = (np.sum(f[Np]*Sp/Hp) - f[p]*np.sum(Sp/Hp)) / ( np.sum(Sp*Hp)/6 )
	
	return lapf

def poissonPressureSolver(network,rho,DUDt,Dir,p0,Neum,gradp):
	
	numPoints = len(network['B'])
	
	A = sparse.lil_matrix((numPoints,numPoints))
	
	b = np.zeros((numPoints,1))
	
	for i in range(numPoints):
		
		if Dir[i]:
			
			A[i,i] = 1
			b[i] = p0[i]
			
		elif Neum[i]:
			
			A[i,i] = 1
			b[i] = 0
			
		else:
			
			Ni = np.asarray(network['N'][i]).reshape(len(network['N'][i]))
			Hi = np.asarray(network['H'][i]).reshape(len(network['H'][i]))
			Si = np.asarray(network['S'][i]).reshape(len(network['S'][i]))
			nHati = np.asarray(network['nHat'][i])
			
			for n in range(len(Ni)):
				A[i,Ni[n]] = (Neum[Ni[n]] - 1) * (Si[n]/Hi[n])
			
			A[i,i] = -np.sum(A[i,:])
			
			bSource = -(rho/2) * np.sum(
				Si * (
					(DUDt[i,0]+DUDt[Ni,0]) * nHati[:,0] +
					(DUDt[i,1]+DUDt[Ni,1]) * nHati[:,1] +
					(DUDt[i,2]+DUDt[Ni,2]) * nHati[:,2]
				)
			)
			
			bNeum = -(1/2) * np.sum(
				Neum[Ni] * Si * (
					(gradp[i,0] + gradp[Ni,0]) * nHati[:,0] +
					(gradp[i,1] + gradp[Ni,1]) * nHati[:,1] +
					(gradp[i,2] + gradp[Ni,2]) * nHati[:,2]
				)
			)
			
			b[i] = bSource + bNeum
	
	
	A = A.tocsc()
	
	M_x = lambda x: sparse.linalg.spsolve(A, x)
	M = sparse.linalg.LinearOperator((numPoints,numPoints), M_x)
	
	p, exitCode = gmres(A, b, M=M, tol=1e-6, maxiter=2000)
	
	return p



























