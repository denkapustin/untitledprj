function main()
%В новых сканах зёрна вылезают за край. Реализовать механихм устранения
%таких объектов. Простое открытие границ по минимальной отсечке не пройдёт,
%так как мелкие зерна, нормально расположенные в центре изображения, могут
%иметь размер меньший этой минимальной отсечки. Посмотреть встроенные
%функции на эту тему. Проблема, возникающая из-за этого: маски с большой
%эрозией и кругом в центре зерна не работают, при вычислении возникают
%ошибки.

% path(path,[pwd filesep 'extractObjFromDir']);
% path(path,[pwd filesep 'labeled']);
% 
% rmpath([pwd filesep 'etxractObjFromDir']);
% rmpath([pwd filesep 'labeled']);
%clb;
ScreenSize = get(0, 'ScreenSize');
%load tmp.mat;
%load tmp2.mat;
%load('tmp.mat','allObjSrcMask', 'paramPrepIm', 'ISrc', 'regionProp');
%load('tmp3nrm.mat');
%load('allImgs.mat','ISrc','allObjSrcMask', 'd', 'regionProp');
%load('tmp3.mat', 'ISrc', 'allObjSrcMask', 'dVec', 'regionProp', 'srcFeats');
%load('allImgs.mat','d31','d32','d33','ISrc','allObjSrcMask', 'd', 'regionProp');
% load('allImgs.mat','d31','d32','d33','srcFeats','featsName');
% dVec = [d31, d32, d33];

% load('allImgs.mat','ISrc','paramPrepIm');
% paramPrepIm2 = {};
% for iter = paramPrepIm
%     paramPrepIm2{end+1} = {iter{:}{1} 12 iter{:}{3} iter{:}{4}};
% end
% paramPrepIm = paramPrepIm2;

load('tmp6.mat');

% d2 = {};
% ISrc2 = {};
% load('tmp.mat','d','ISrc');
% d2 = {d2{:}, d{:}};
% ISrc2 = {ISrc2{:}, ISrc{:}};
% load('tmp2.mat','dVec','ISrc','regionProp');
% d = vec2cell(dVec,regionProp);
% d2 = {d2{:}, d{:}};
% ISrc2 = {ISrc2{:}, ISrc{:}};
% load('tmp3.mat','dVec','ISrc','regionProp');
% d = vec2cell(dVec,regionProp);
% d2 = {d2{:}, d{:}};
% ISrc2 = {ISrc2{:}, ISrc{:}};
% d = d2;
% ISrc = ISrc2;
% save('allImgs.mat','d','ISrc','-append','-v7.3');
% pause;

% for i=1:10*20
%     subplottight(10,20,i);
%     imshow(srcImgs{i},'border','tight');
% end

% %Исходное изображение
% ISrc = {};
% path = [pwd filesep 'dirSource' filesep];
% files = dir([path '*.bmp']);
% for file = files'
%     ISrc{end+1} = imread([path file.name]);
% end
% 

% %Маски ячеек для ячеистого изображения со сканера
% scanMask = {};
% for img = ISrc
%     scanMask{end+1} = getMask(img{:});
% end

% %Исходное изображение со сканерной маской
% IScanMask = {};
% for k = 1:numel(ISrc)
%     IMask = scanMask{k}.im{1};
%     if ( numel(scanMask{k}.im) > 1)
%         for imask = scanMask{k}.im'
%             IMask = IMask + imask{:};
%         end
%     end
%     IMask = uint8(repmat(~logical(IMask), [1 1 3]))*255;
%     IScanMask{end+1} = ISrc{k} + IMask;
% end

% %Маска для всех объектов изображения. Начальная. Erode 6
% allObjSrcMask = {};
% paramPrepIm = repmat({{NaN, 6, NaN, NaN}}, [1,numel(ISrc)]);
% paramPrepImOut = {};
% for u = 1:numel(ISrc)
%     [BW, bgThrsh, erode, minAreaThrsh, maxAreaThrsh] = prepIm(IScanMask{u}, 'BackgroundThreshold', paramPrepIm{u}{1}, 'Erode', paramPrepIm{u}{2}, 'minAreaThreshold', paramPrepIm{u}{3}, 'maxAreaThreshold', paramPrepIm{u}{4}, 'Show', 0);
%     allObjSrcMask{end+1} = BW;
%     paramPrepImOut{end+1} = {bgThrsh erode minAreaThrsh maxAreaThrsh};
% end
% paramPrepIm = paramPrepImOut;

% figure('NumberTitle', 'off', 'toolbar', 'none');
% for i = 1:numel(IScanMask)
%     imshow(allObjSrcMask{i}, 'border', 'tight');
%     hold on;
%     imshow(IScanMask{i}, 'border', 'tight');
%     alpha .5;
%     hold off;
% end

% %Маска для всех объектов изображения. Erode 22|25
% allObjErode22Mask = {};
% for mask = allObjSrcMask;
%     allObjErode22Mask{end+1} = prepIm(mask{:}, 'Erode', 22, 'minAreaThreshold', 30);
% end

% figure('NumberTitle', 'off', 'toolbar', 'none');
% for i = 1:numel(IScanMask)
%     imshow(allObjErode22Mask{i}, 'border', 'tight');
%     hold on;
%     imshow(IScanMask{i}, 'border', 'tight');
%     alpha .5;
%     hold off;
% end

% %Маска для всех объектов изображения. Диск по центру зерна
% %regionProp для всех объектов для начальной маски.
% allObjCircleMask = {};
% regionProp = {};
% for mask = allObjSrcMask
%     rp = regionprops(bwlabel(mask{:}),'all');
%     regionProp{end+1} = rp;
%     %Если закоментить строчку ниже, то визуализируя mask{:} будет видно
%     границы круга в зерне.
%     mask{:}(:) = 0;
%     for obj = rp'
%         circle = strel('disk', round(obj.MinorAxisLength/2), 0).getnhood;
%         cx = obj.Centroid(1);
%         cy = obj.Centroid(2);
%         d = size(circle,1);
%         %Ноль всего изображения слева вверху. Растёт вправо и вниз.
%         topleftx = round(cx - d/2);
%         toplefty = round(cy - d/2);
%         for i = toplefty:toplefty + d -1 
%             for j = topleftx:topleftx + d - 1
%                mask{:}(i,j) = 200*circle(1 + i - toplefty, 1 + j - topleftx); 
%             end
%         end
%     end
%     allObjCircleMask{end+1} = mask{:};
% end

% figure('NumberTitle', 'off', 'toolbar', 'none');
% for i = 1:numel(IScanMask)
%     imshow(allObjCircleMask{i}, 'border', 'tight');
%     hold on;
%     imshow(IScanMask{i}, 'border', 'tight');
%     alpha .5;
%     hold off;
% end

% %Проведём разметку
% dNew = {};
% for iterImg = 1:numel(ISrc)
%     fig = figure('toolbar', 'none', 'numberTitle', 'off');
%     imshow(ISrc{iterImg} .* uint8(repmat(allObjSrcMask{iterImg}, [1 1 3])), 'border', 'tight');
%     moveWnd(fig,get(0,'ScreenSize'),[2 1 mod(iterImg+1,2)+1]);
%     %dNew{end+1} = labeled(fig, bwlabel(allObjSrcMask{iterImg}), regionProp{iterImg}, 2, d{iterImg});
%     dNew{end+1} = labeled(fig, bwlabel(allObjSrcMask{iterImg}), regionProp{iterImg}, 3, d{iterImg});
%     for class = 0:max(dNew{end})
%         set(fig, 'name', sprintf('%s %i:%i', get(fig,'name'), class, numel(find(dNew{end}==class))));
%     end    
% end
% dVecNew = [dNew{:}];
% pause;

% %Устраним дефект сканера(неравномерное затенение справа от слайдовой лампы)
% ISrcNrm = {};
% for iter = 1:numel(ISrc)
%     nrm = nrmBckg(ISrc{iter}, allObjSrcMask{iter}, paramPrepIm{iter}{1});
%     ISrcNrm{end+1} = ISrc{iter} + nrm;
% end
% ISrcOrig = ISrc;
% ISrc = ISrcNrm;

% %Получим наборы изображений объектов под тремя видами маск(исходная, с кругом в
% %центре,с эрозией границы в 25 пикселей.
% imgsSrcMask = {};
% % imgsErode22Mask = {};
% % imgsCircleMask = {};
%  for iter = 1:numel(ISrc)
%     imgsSrcMask{end+1} = extractObjFromImage(ISrc{iter}, 'BWImg', allObjSrcMask{iter});
% %     imgsErode22Mask{end+1} =  extractObjFromImage(ISrc{iter}, 'BWImg', allObjErode22Mask{iter});
% %     imgsCircleMask{end+1} = extractObjFromImage(ISrc{iter}, 'BWImg', allObjCircleMask{iter});
%  end
% srcImgs = [imgsSrcMask{:}];
% % circleImgs = [imgsCircleMask{:}];
% % erode22Imgs = [imgsErode22Mask{:}];
% % %Можно вывести все объекты.

% AllImgsSrcMask = {};
% for i = imgsSrcMask
%     AllImgsSrcMask = [AllImgsSrcMask{:} i{:}]; 
% end
% AllImgsErode22Mask = {};
% for i = imgsErode22Mask
%     AllImgsErode22Mask = [AllImgsErode22Mask{:} i{:}];
% end
% AllImgsCircleMask = {};
% for i = imgsCircleMask
%     AllImgsCircleMask = [AllImgsCircleMask{:} i{:}];
% end

% %Вычсилим все признаки интенсивности для всех объектов.
% b(1).name = 'basicint';
% b(1).options.show = 0;
% options.b = b;
% options.colstr = 'rgb';
% srcFeats = [];
% for obj = srcImgs
%     [Xi, Xn] = Bfx_int(obj{:}, [],options);
%     srcFeats = [srcFeats; Xi];    
% end
% featsName = Xn;

% circleFeats = [];
% for obj = circleImgs
%     [Xi, Xn] = Bfx_int(obj{:}, [],options);
%     circleFeats = [circleFeats; Xi];    
% end
% 
% erode25Feats = [];
% for obj = erode25Imgs
%     [Xi, Xn] = Bfx_int(obj{:}, [], options);
%     erode25Feats = [erode25Feats; Xi];
% end

% %Определим рекомендуемый Balu лучший классификатор и набор признаков.
% [indTrain, indTest] = getRndInd(numel(dVec));
% dTrain = dVec(indTrain)';
% dTest = dVec(indTest)';

%  bcl(1).name = 'knn';   bcl(1).options.k = 3;      % KNN with 3 neighbors
%  bcl(2).name = 'knn';   bcl(2).options.k = 5;      % KNN with 5 neighbors
%  bcl(3).name = 'knn';   bcl(3).options.k = 7;      % KNN with 7 neighbors
%  bcl(4).name = 'knn';   bcl(4).options.k = 9;      % KNN with 9 neighbors
% bcl(5).name = 'lda';   bcl(5).options.p = [];     % LDA
% bcl(6).name = 'qda';   bcl(6).options.p = [];     % QDA
% bcl(7).name = 'maha';  bcl(7).options = [];       % Euclidean distance
% bcl(8).name = 'adaboost'; bcl(8).options.iter = 10;
% bcl(9).name = 'libsvm'; bcl(9).options.kernel = '-q 0 -t 0 -v 10 -c 1';
% bcl(10).name = 'libsvm'; bcl(10).options.kernel = '-q 0 -t 0 -v 10 -c 1.5';
% bcl(11).name = 'libsvm'; bcl(11).options.kernel = '-q 0 -t 0 -v 10 -c 2';
% bcl(12).name = 'libsvm'; bcl(12).options.kernel = '-q 0 -t 0 -v 10 -c 4';
% bcl(13).name = 'libsvm'; bcl(13).options.kernel = '-q 0 -t 0 -v 10 -c 8';
% bcl(14).name = 'dmin';  bcl(14).options = [];       % Mahalanobis distance
% bcl(15).name = 'boostVJ'; bcl(15).options.iter = 25;
% bcl(16).name = 'nnglm'; bcl(16).options.method = 1; bcl(16).options.iter = 10;
% bcl(17).name = 'nnglm'; bcl(17).options.method = 2; bcl(17).options.iter = 10;
% bcl(18).name = 'nnglm'; bcl(18).options.method = 3; bcl(18).options.iter = 10;
% 
% 
% bfs(1).name = 'sfs';   bfs(1).options.b.name    = 'fisher';
% bfs(2).name = 'sfs';   bfs(2).options.b.name    = 'knn'; bfs(2).options.b.options.k = 5;
% % bfs(3).name = 'sfs';   bfs(3).options.b.name    = 'sp100';
% % bfs(4).name = 'sfs';   bfs(4).options.b.name    = 'knn'; bfs(4).options.b.options.k = 7;
% % bfs(5).name = 'rank';  bfs(5).options.criterion = 'roc';
% 
% optionsBCL.Xn   = featsName;
% optionsBCL.bcl  = bcl;
% optionsBCL.bfs  = bfs;
% optionsBCL.m    = 10;
% optionsBCL.show = 0;
% 
% res = [];
%             % Подбор оптимальных настроек для libSVM классификатора. Начало
%             % feats = {[1 2], [1 2 7 8 13 14], [2 4 7 8 9 13 15 16], [2 13], [2 13 14], [2 3 4 13], [2 13 14 8], [2 3 4 8 13], [2 8 13 14 15], [2 3 4 8 13 16]};
%             % cgcv = [];
%             % tic;
%             % for f = feats
%             %     [t d r c g cv] = bestcv(srcFeats(:,f{:}),dVec');
%             %     cgcv = [cgcv; t d r c g cv];
%             % end
%             % toc;
%             % [del, srt] = sort(cgcv(:,6));
%             % sortcv = cgcv(srt,:);
%             % Подбор оптимальных настроек для libSVM классификатора. Конец
% tic;
% for iter = 1:6
%     bclr = bcl;
%     %Для выбранного классификатора проведём ещё несколько итераций с
%     %целью возможного получения набора других рекомендуемых признаков.
%     for iter2 = 1:3
%         [bcs, selec] = Bcl_balu(srcFeats, dVec', bcl,bfs,optionsBCL);
%         op.b = bcs; op.v = 10; op.show = 0; op.c = 0.95;
%         [p,ci] = Bev_crossval(srcFeats(:,selec),dVec',op);
%         if isempty(bcs.options)
%             res = [res; [bcs.name {''}], selec, p];
%         else
%             res = [res; [bcs.name {struct2cell(bcs.options)}], selec, p];
%         end
%         bcl = bcs;
%     end
%     bcl = bclr;
% end
% toc;
% %res - без сортировки. Группировка по классификаторам.
% [del, srt] = sort([res{:,4}]); 
% result = res(fliplr(srt)',:);% Сортировка по показателю точности

%Для выбранных вручную наборов признаков полным перебором определим лучший
%классификатор. 
%allFeats = {srcFeats, circleFeats, erode25Feats};
allFeats = {srcFeats};
numbExp = 32;
title = ['Mask', 'Classifier', 'Features', 'Mean', 'Median', 'std', 'min', 'max', repmat({'Exprm'}, [1 numbExp])];
performance = {};
%descMask = {'Orig' 'Circle' 'Erode25'};
descMask = {'Orig'};
descFeats = {'2 13 14', '1 2', '1 2 7 8 13 14',...
    '2 4 3 13', '2 4', '2 7', '2 13', '2 7 13', '2 4 3',...
    '1 6 13', '1 6 14', '1 7 13', '1 7 14', '2 6 13', '2 6 14', '2 7 14',...
    '2 4 16 7 15', '2 4 16 7 15 13 9 8', '2 4 7 15', '2 8 4 16 10 15', '2 14 4 13 7', '2 14 4 13'};
knn9.name = 'knn';
knn9.desc = 'knn9';
knn9.options.k = 9;

knn5.name = 'knn';
knn5.desc = 'knn5';
knn5.options.k = 5;

lda.name = 'lda';
lda.desc = lda.name;
lda.options.p = [];

qda.name = 'qda';
qda.desc = qda.name;
qda.options.p = [];

maha.name = 'maha';
maha.desc = maha.name;
maha.options = [];

adaboost.name = 'adaboost';
adaboost.desc = adaboost.name;
adaboost.options.iter = 10;

nnglm2.name = 'nnglm';
nnglm2.desc = 'nnglm2'; 
nnglm2.options.method = 2;
nnglm2.options.iter = 10;

nnglm3.name = 'nnglm';
nnglm3.desc = 'nnglm3'; 
nnglm3.options.method = 3;
nnglm3.options.iter = 10;

libsvm1.name = 'libsvm';
libsvm1.desc = 'libsvm linear c1';
libsvm1.options.kernel = '-q 0 -t 0 -v 10 -c 1';

libsvm2.name = 'libsvm';
libsvm2.desc = 'libsvm linear c2';
libsvm2.options.kernel = '-q 0 -t 0 -v 10 -c 2';

libsvm4.name = 'libsvm';
libsvm4.desc = 'libsvm linear c4';
libsvm4.options.kernel = '-q 0 -t 0 -v 10 -c 4';

libsvm8.name = 'libsvm';
libsvm8.desc = 'libsvm linear c8';
libsvm8.options.kernel = '-q 0 -t 0 -v 10 -c 8';

dmin.name = 'dmin';
dmin.desc = dmin.name;
dmin.options = [];

boostvj.name = 'boostVJ';
boostvj.desc = boostvj.name;
boostvj.options.iter = 25;

vekClassifiers = {knn9, knn5, lda, qda, maha, ...
   adaboost, nnglm2, nnglm3, libsvm1, libsvm2, libsvm4, libsvm8, dmin};%, boostvj};

for i=1:numel(descMask)
    for j=1:numel(vekClassifiers)
        for k=1:numel(descFeats)
            performance(end+1, :) = {descMask{i} vekClassifiers{j}.desc descFeats{k}};
        end
    end
end

perf = [];
tic;
for numb = 1:numbExp;
    %[indTrain, indTest] = getRndInd(numel(dVec));
    [indTrain, indTest] = Bds_ixstratify(dVec',0.5);
    dTrain = dVec(indTrain)';
    dTest = dVec(indTest)';

    p = [];
    for i=1:numel(descMask)
        XFeats = allFeats{i};
        XTrain = XFeats(indTrain,:);
        XTest = XFeats(indTest,:);
        for j=1:numel(vekClassifiers)
            for k=1:numel(descFeats)
                p = [p; clcPerf(j, k, vekClassifiers, descFeats, XTrain, XTest, dTrain, dTest)];
            end
        end
    end
    perf = [perf p];    
end
toc;

mn = mean(perf');
[del, srt] = sort(mn);
performance = [performance num2cell(mn') num2cell(median(perf')') num2cell(std(perf')') num2cell(min(perf')') num2cell(max(perf')') num2cell(perf)];
performanceSort = performance(fliplr(srt)',:);
performance = [title; performance];
performanceSort = [title; performanceSort];
pause;

% [indTrain, indTest] = getRndInd(numel(dVec));
% dTrain = dVec(indTrain)';
% dTest = dVec(indTest)';
% for k = 1:numel(descMask)
%     IAll = AllImgs{k};
%     ITrain = IAll(indTrain);
%     dTrain = dVec(indTrain)';
%     ITest = IAll(indTest);
%     dTest = dVec(indTest)';
% 
%     b(1).name = 'basicint';
%     b(1).options.show = 0;
%     options.b = b;
%     options.colstr = 'rgb';
%     XTrain = [];
%     for obj = ITrain
%         [Xi, Xn] = Bfx_int(obj{:},[],options);
%         XTrain = [XTrain; Xi];
%     end
%     XTest = [];
%     for obj = ITest
%         [Xi,Xn] = Bfx_int(obj{:},[],options);
%         XTest = [XTest; Xi];
%     end
% 
%     bfs = Bfs_build({'all'});
%     bcl = Bcl_build({'knn10', 'svm1', 'lda'});
%     opt.Xn  = Xn;
%     opt.m   = 3;
%     opt.v   = 10;
%     [bcs, selec] = Bcl_balu(XTrain,dTrain,bcl,bfs,opt);
% 
%     op.kernel = 1;%4;
%     op.p = [];
%     op.k = 10;
%     %sel = [1:18];
%     %sel = [1,2,7,8,13,14];
%     %sel = [2,4,16,7,15];
%     %sel = [2,4,16,7,15,13,9,8];
%     sel = [1,2];
%     XTrain = XTrain(:,sel);
%     XTest = XTest(:,sel);
%     ds = Bcl_svmplus(XTrain, dTrain, XTest,op);
%     disp(sprintf('SVMPlus:%2.2f', 100*Bev_performance(ds,dTest)));
%     ds = Bcl_knn(XTrain, dTrain, XTest, op);
%     disp(sprintf('KNN%d:%2.2f', op.k, 100*Bev_performance(ds,dTest)));
%     ds = Bcl_lda(XTrain, dTrain, XTest, op);
%     disp(sprintf('LDA:%2.2f', 100*Bev_performance(ds,dTest)));
%     %Bio_plotfeatures(XTrain, dTrain);
%     %p = Bev_performance(ds,dTest);
% 
%     XAllByFileImg = {};
%     for k =1:length(ISrc)
%         X = [];
%         for obj = imgsSrcMask{k};
%             [Xi, Xn] = Bfx_int(obj{:}, [],options);
%             X = [X; Xi];
%         end
%         XAllByFileImg{end+1} = X;
%     end
%     op = Bcl_lda(XTrain, dTrain, op);
%     for k = 1:numel(ISrc)
%         f1 = figure('numberTitle', 'off', 'toolbar', 'none');
%         imshow(ISrc{k} .* uint8(repmat(allObjSrcMask{k}, [1 1 3])), 'border', 'tight');
%         f2 = figure('numberTitle', 'off', 'toolbar', 'none');
%         imshow(ISrc{k} .* uint8(repmat(allObjSrcMask{k}, [1 1 3])), 'border', 'tight');        
%         compareLabels(f1, f2, regionProp{k}, d{k}, Bcl_lda(XAllByFileImg{k}(:,sel),op)');
%     end     
%end

function p = clcPerf(indClassifier, indSlcFeats, vecClassifier, vecSlcFeats, XTrain, XTest, dTrain, dTest)
%vecFeatures - вычисленные признаки для i-ой маски.
%vecSlcFeats - вектор с номерами отобранных признаков.
    slcFeats = str2num(vecSlcFeats{indSlcFeats});

    XTrain = XTrain(:, slcFeats);
    XTest = XTest(:, slcFeats);
    
    classifier = vecClassifier{indClassifier};
    ds = feval(['Bcl_' classifier.name], XTrain, dTrain, XTest, classifier.options);
    p = Bev_performance(ds, dTest);

% function p = clcPerf(indMask, indClassifier, indSlcFeats, vecFeatures, vecClassifier, vecSlcFeats, dTrain, dTest, indTrain, indTest)
% %vecFeatures - вычисленные признаки для i-ой маски.
% %vecSlcFeats - вектор с номерами отобранных признаков.
%     Features = vecFeatures{indMask};
%     slcFeats = str2num(vecSlcFeats{indSlcFeats});
% 
%     XTrain = Features(indTrain, slcFeats);
%     XTest = Features(indTest, slcFeats);
%     
%     classifier = vecClassifier{indClassifier};
%     ds = feval(['Bcl_' classifier.name], XTrain, dTrain, XTest, classifier.options);
%     p = Bev_performance(ds, dTest);
% clb;
% % s = [30:0.25:60]';
% % t = Bcl_svmplus(XTrain, dTrain, s,op);
% % tmp =  find(t == 1);
% % tmp2 = tmp(1);
% % thrsh = s(tmp2)
% 
% % XTrain = XTrain(:,[7 3 4]);
% % XTest = XTest(:,[7 3 4]);
% % figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Mychnue'), histfit(XTrain(find(dTrain == 0)));
% % figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Steklovidnue'), histfit(XTrain(find(dTrain == 1)));
% % f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Mychnue Steklovidnue');
% % histfit(XTrain(find(dTrain == 0)));
% % set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% % hold on;
% % histfit(XTrain(find(dTrain == 1)));
% % set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% % 
% % figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Test Mychnue'), histfit(XTest(find(dTest == 0)));
% % figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Test Steklovidnue'), histfit(XTest(find(dTest == 1)));
% % f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Test Mychnue Steklovidnue');
% % histfit(XTrain(find(dTest == 0)));
% % set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% % hold on;
% % histfit(XTrain(find(dTest == 1)));
% % set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% % 
% % %figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Mychnue'), histfit([XTrain(find(dTrain == 0)); XTest(find(dTest == 0))]);
% % %figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Steklovidnue'), histfit([XTrain(find(dTrain == 1)); XTest(find(dTest == 1))]);
% % 
% % f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Mychnue');
% % histfit(XTrain(find(dTrain == 0)));
% % set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% % hold on;
% % histfit(XTest(find(dTest == 0)));
% % set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% % 
% % f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Steklovidnue');
% % histfit(XTrain(find(dTrain == 1)));
% % set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% % hold on;
% % histfit(XTest(find(dTest == 1)));
% % set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% % 
% % f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Mychnue Steklovidnue');
% % XAll = [XTrain; XTest];
% % dAll = [dTrain; dTest];
% % histfit(XAll(find(dAll == 0)));
% % set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% % hold on;
% % histfit(XAll(find(dAll == 1)));
% % set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% 
% figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Mychnue');
% [f,x] = hist(XTrain(find(dTrain == 0)));
% bar(x,f/numel(XTrain(find(dTrain == 0))));
% figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Steklovidnue');
% [f,x] = hist(XTrain(find(dTrain == 1)));
% bar(x,f/numel(XTrain(find(dTrain == 1))));
% f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Mychnue Steklovidnue');
% [f,x] = hist(XTrain(find(dTrain == 0)));
% bar(x,f/numel(XTrain(find(dTrain == 0))));
% set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% hold on;
% [f,x] = hist(XTrain(find(dTrain == 1)));
% bar(x,f/numel(XTrain(find(dTrain == 1))));
% set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% 
% figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Test Mychnue');
% [f,x] = hist(XTest(find(dTest == 0)));
% bar(x,f/numel(XTest(find(dTest == 0))));
% figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Test Steklovidnue');
% [f,x] = hist(XTest(find(dTest == 1)));
% bar(x,f/numel(XTest(find(dTest == 1))));
% f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Test Mychnue Steklovidnue');
% [f,x] = hist(XTest(find(dTest == 0)));
% bar(x,f/numel(XTest(find(dTest == 0))));
% set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% hold on;
% [f,x] = hist(XTest(find(dTest == 1)));
% bar(x,f/numel(XTest(find(dTest == 1))));
% set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% 
% f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Mychnue');
% [f,x] = hist(XTrain(find(dTrain == 0)));
% bar(x,f/numel(XTrain(find(dTrain == 0))));
% set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% hold on;
% [f,x] = hist(XTest(find(dTest == 0)));
% bar(x,f/numel(XTest(find(dTest == 0))));
% set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% 
% f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Steklovidnue');
% [f,x] = hist(XTrain(find(dTrain == 1)));
% bar(x,f/numel(XTrain(find(dTrain == 1))));
% set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% hold on;
% [f,x] = hist(XTest(find(dTest == 1)));
% bar(x,f/numel(XTest(find(dTest == 1))));
% set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);
% 
% f = figure('numberTitle', 'off', 'toolbar', 'none', 'name', 'Train Test Mychnue Steklovidnue');
% XAll = [XTrain; XTest];
% dAll = [dTrain; dTest];
% [f,x] = hist(XAll(find(dAll == 0)));
% bar(x,f/numel(XAll(find(dAll == 0))));
% set(findobj(gca,'Type','patch'), 'FaceColor', [1 0 0]);
% hold on;
% [f,x] = hist(XAll(find(dAll == 1)));
% bar(x,f/numel(XAll(find(dAll == 1))));
% set(findobj(gca,'Type','patch'), 'FaceAlpha', 0.5);