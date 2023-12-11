import numpy as np




def k_means_init(input,num_clusters):
    centers=[]
    for i in range(num_clusters):
        i=np.random.random(np.size(input[0]))
        centers.append(i)
    return centers
def classify(input,centers,num_clusters):
    clusters=[[] for i in range(num_clusters)]
    num_list=[i for i in range(num_clusters)]
    records = [[] for i in range(num_clusters)]
    for label,i in enumerate(input):
        distants=[]
        for center in centers:
            distant=np.linalg.norm(i-center)+1/(np.sum(i*center)+0.1)
            distants.append(distant)
        min_distant = min(distants)
        for j ,distant in enumerate(distants):
            if distant == min_distant:
                clusters[j].append(i)
                records[j].append(label)
                break
    records_dict=dict(zip(num_list,records))
    return clusters,records_dict
def centers_refresh(clusters,input):
    centers = []
    for cluster in clusters:
        if len(cluster):
            center=sum(cluster)/len(cluster)
        else:
            center =np.random.random(np.size(input[0]))
        centers.append(center)
    return centers

def judge(centers,pre_centers):
    for i in range(len(centers)):
        if (centers[i] == pre_centers[i]).all():
            pass
        else:
            return False
    return  True
def k_means(input,num_clusters,iter_number):
    centers=k_means_init(input,num_clusters)
    for i in range(iter_number):
        pre_centers=centers
        clusters,records=classify(input,centers,num_clusters)
        centers=centers_refresh(clusters,input)
        if judge(centers,pre_centers):
            centers=np.array(centers)
            return  records,centers
    centers = np.array(centers)
    return records,centers