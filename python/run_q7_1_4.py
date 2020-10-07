import torchvision.datasets

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
emnist = torchvision.datasets.EMNIST(root='../data', split='balanced', download=True)

print(emnist)