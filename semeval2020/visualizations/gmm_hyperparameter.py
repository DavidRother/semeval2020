import numpy as np

# n_components
rdata_10 = [0.19995214217001692, 0.18232790074365854, 0.19995258283763587, 0.1895883825291255, 0.18999964289477234,
            0.19330400896477692, 0.22969312921571422, 0.21612970301166823, 0.17021393166021526, 0.21250473389985794]

rdata_05 = [0.17992712812807984, 0.24119230586442622, 0.20249869494623302, 0.20537709393133155, 0.23614974379652356,
            0.17431376159213902, 0.176630968209319, 0.17792556797177592, 0.16203203460830742, 0.17591301290255107,
            0.16725183968687396, 0.23801551916182165]

rdata_03 = [0.16819052847959462, 0.27609329420383194, 0.19433042512914145, 0.14990505996955827, 0.23606097976472776,
            0.26464894424356206, 0.1863747333729308, 0.27817184442511744, 0.24362824973484346, 0.16402259543891426,
            0.18669186048337527, 0.20869592878621052]

adata_10 = [0.5014349026445801, 0.5162843286835223, 0.5165531458878233, 0.5120922333623946, 0.5524456916594014,
            0.4866445074106365, 0.5035463891310665, 0.5275047224643998, 0.49179108544027894, 0.532416085440279]

adata_05 = [0.5555316405114792, 0.5212211203138623, 0.5550276082534147, 0.5781540613193838, 0.5415168192385934,
            0.534227877070619, 0.5516392400464981, 0.5342251525719268, 0.5086929671607091, 0.5066069093286836,
            0.5277063353676257, 0.5201767291485033]

adata_03 = [0.5266365155478059, 0.5422587910491136, 0.5773757628596339, 0.5565760316768381, 0.5521378233071781,
            0.5654133972682359, 0.5659174295263005, 0.5399093650101714, 0.5321082170880558, 0.5222319093286836,
            0.5269026082534147, 0.5513676983435047]

print(np.mean(rdata_03), np.std(rdata_03))
print(np.mean(rdata_05), np.std(rdata_05))
print(np.mean(rdata_10), np.std(rdata_10))
print(np.mean(adata_03), np.std(adata_03))
print(np.mean(adata_05), np.std(adata_05))
print(np.mean(adata_10), np.std(adata_10))
