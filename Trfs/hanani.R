library(edgeR) ; library(ggplot2) ; library(matrixStats)

hnni<-read.csv('hanani.csv')
trfs<-read.csv('tRNA_Exclusive_Combined_data.csv') ; rownames(trfs)<-trfs$X ; trfs<-trfs[,-1]
colnames(trfs)<-unlist(lapply(colnames(trfs),function(x) strsplit(as.character(x),'X')[[1]][2]))
cpm_trfs<-as.data.frame(cpm(trfs))
meta<-read.csv('trf_meta.csv')[,2:5] ; colnames(meta)<-c('trf','seq','type','details')
meta$len<-unlist(lapply(meta$trf,function(x) strsplit(as.character(x),'-')[[1]][2]))
meta$trna<-unlist(lapply(meta$details,function(x) substr(strsplit(as.character(x),'_')[[1]][2],1,3)))
meta$codon<-unlist(lapply(meta$details,function(x) substr(strsplit(as.character(x),'_')[[1]][2],4,6)))
meta$gene<-unlist(lapply(meta$details,function(x) strsplit(as.character(x),'_')[[1]][1]))
meta$origin<-factor(meta$gene %in% c('MT','trnalookalike8'),labels=c('Nuclear','Mitochondrial'))

tgg7d<-cpm_trfs[,colnames(cpm_trfs) %in% subset(hnni$id,hnni$gngl=='T' & hnni$hours==168)]
tgg7d<-tgg7d[rowMedians(as.matrix(tgg7d))>10,]

pvTab<-as.data.frame(t(data.frame(row.names=c('tRF','log2FC','p'))))
for(t in rownames(tgg7d)){
  tmp<-data.frame(id=colnames(tgg7d),exp=as.numeric(tgg7d[t,])) ; tmp<-merge(tmp,hnni,by='id')
  tt<-t.test(tmp$exp~tmp$trt)
  pvTab<-rbind(pvTab,data.frame('tRF'=t,'log2FC'=log2(tt$estimate[2]/tt$estimate[1]),'p'=tt$p.value))
} ; pvTab$fdr<-p.adjust(pvTab$p,'fdr')

ggplot(pvTab,aes(log2FC,-log10(p),col=p<0.05))+theme_classic()+geom_point()


scg7d<-cpm_trfs[,colnames(cpm_trfs) %in% subset(hnni$id,hnni$gngl=='S' & hnni$hours==168)]
scg7d<-scg7d[rowMedians(as.matrix(scg7d))>10,]

pv2Tab<-as.data.frame(t(data.frame(row.names=c('tRF','log2FC','p'))))
for(t in rownames(scg7d)){
  tmp<-data.frame(id=colnames(scg7d),exp=as.numeric(scg7d[t,])) ; tmp<-merge(tmp,hnni,by='id')
  tt<-t.test(tmp$exp~tmp$trt)
  pv2Tab<-rbind(pv2Tab,data.frame('tRF'=t,'log2FC'=log2(tt$estimate[2]/tt$estimate[1]),'p'=tt$p.value))
} ; pv2Tab$fdr<-p.adjust(pv2Tab$p,'fdr')

ggplot(pv2Tab,aes(log2FC,-log10(p),col=p<0.05))+theme_classic()+geom_point()


##### edgeR #####

# Tganglion
cts<-trfs[,colnames(trfs) %in% subset(hnni$id,hnni$gngl=='T' & hnni$hours==168)] 
cld<-subset(hnni,hnni$gngl=='T' & hnni$hours==168) ; cts<-cts[,as.character(cld$id)] 
cts<-cts[,order(cld$id)] ; cld<-cld[order(cld$id),] ; nrow(cld)==sum(cld$id==colnames(cts))
y <- DGEList(counts=cts,group=cld$trt) ; keep <- filterByExpr(y) ; y <- y[keep, , keep.lib.sizes=FALSE]
y$samples$lib.size <- colSums(y$counts) ; y <- calcNormFactors(y)

cts1<-as.data.frame(cpm(y,log = F)) # creates the normalized counts matrix
dsgn <- model.matrix(~sex+trt, data = cld) ; y <- estimateDisp(y, dsgn, robust = T) ; head(dsgn,3)
fit <- glmQLFit(y, dsgn, robust = T) ; lrt <- glmLRT(fit,coef = 3) 

toRemove<-c() ; for(g in rownames(cts1)){
  if(quantile(as.numeric(cts1[g,]),prob=0.85)<mean(as.numeric(cts1[g,]))){toRemove<-append(toRemove,g)}}
sg_tGens<-as.data.frame(topTags(lrt,adjust.method = 'fdr',n = nrow(cts1)))
sg_tGens$log2FC<-log2(2.718^sg_tGens$logFC)
sg_tGens$isSg<-factor(as.numeric(sg_tGens$FDR<0.051),labels = c('N.S.','Sgnificant'))
sg_tGens$trf<-rownames(sg_tGens) ; sg_tGens<-merge(sg_tGens,meta,by='trf')
# write.csv(sg_tGens,'EdgeR_T_ganglion.csv')

ggplot(sg_tGens,aes(log2FC,-log10(PValue),shape=FDR<0.05,col=type))+theme_classic()+geom_point()

g<-'tRF-18-W9RKM80E'
tmp<-data.frame(id=colnames(cpm_trfs),exp=as.numeric(cpm_trfs[g,])) ; tmp<-merge(tmp,hnni,by='id')
ggplot(tmp,aes(as.factor(hours),exp,col=trt))+theme_classic()+facet_wrap(~gngl)+
  geom_point(position=position_jitterdodge(jitter.width=0.2,jitter.height=0))


# S ganglion
cts<-trfs[,colnames(trfs) %in% subset(hnni$id,hnni$gngl=='S' & hnni$hours==168)] 
cld<-subset(hnni,hnni$gngl=='S' & hnni$hours==168) ; cts<-cts[,as.character(cld$id)] 
cts<-cts[,order(cld$id)] ; cld<-cld[order(cld$id),] ; nrow(cld)==sum(cld$id==colnames(cts))
y <- DGEList(counts=cts,group=cld$trt) ; keep <- filterByExpr(y) ; y <- y[keep, , keep.lib.sizes=FALSE]
y$samples$lib.size <- colSums(y$counts) ; y <- calcNormFactors(y)

cts1<-as.data.frame(cpm(y,log = F)) # creates the normalized counts matrix
dsgn <- model.matrix(~sex+trt, data = cld) ; y <- estimateDisp(y, dsgn, robust = T) ; head(dsgn,3)
fit <- glmQLFit(y, dsgn, robust = T) ; lrt <- glmLRT(fit,coef = 3) 

toRemove<-c() ; for(g in rownames(cts1)){
  if(quantile(as.numeric(cts1[g,]),prob=0.85)<mean(as.numeric(cts1[g,]))){toRemove<-append(toRemove,g)}}
sg_sGens<-as.data.frame(topTags(lrt,adjust.method = 'fdr',n = nrow(cts1)))
sg_sGens$log2FC<-log2(2.718^sg_sGens$logFC)
sg_sGens$isSg<-factor(as.numeric(sg_sGens$FDR<0.051),labels = c('N.S.','Sgnificant'))
sg_sGens$trf<-rownames(sg_sGens) ; sg_sGens<-merge(sg_sGens,meta,by='trf')
# write.csv(sg_sGens,'EdgeR_S_ganglion.csv')

ggplot(sg_sGens,aes(log2FC,-log10(PValue),shape=FDR<0.05,col=type))+theme_classic()+geom_point()

g<-'tRF-18-W9RKM80E'
tmp<-data.frame(id=colnames(cpm_trfs),exp=as.numeric(cpm_trfs[g,])) ; tmp<-merge(tmp,hnni,by='id')
ggplot(tmp,aes(as.factor(hours),exp,col=trt))+theme_classic()+facet_wrap(~gngl)+
  geom_point(position=position_jitterdodge(jitter.width=0.2,jitter.height=0))
