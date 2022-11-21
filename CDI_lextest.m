function recall = CDI_lextest(audiodir,embdir,opmode,runparallel)
% function CDI_lextest(audiodir,embdir,opmode,runparallel)
%
% Inputs:
%
%   audiodir        : location of the CDI audiofiles (.wav)
%   embdir          : location of corresponding embeddings (.txt)
%   opmode          : 'single' (default) for one embedding per .wav
%                     'full' for frame-level embeddings
%   runparallel     : 0 (default) / 1. Use MATLAB parallel pool? Only
%                     applies to opmode = 'full'.
%   
% Outputs:
%   recall          : overall separability score [0,1]. Larger is better.   

if nargin <4
    runparallel = 0;
end

if nargin <3
    opmode = 'single';
end

%audiodir = '/Users/rasaneno/speechdb/CDI_synth/';

tmp = dir([audiodir '/*.wav']);

filenames = cell(length(tmp),1);
word = cell(length(tmp),1);
spkr = cell(length(tmp),1);
style = cell(length(tmp),1);
for k = 1:length(tmp)
    filenames{k} = [audiodir '/' tmp(k).name];
    word{k} = tmp(k).name(1:end-7);
    spkr{k} = tmp(k).name(end-5);
    if(strcmp(tmp(k).name(end-4),'q'))
        style{k} = 'Q';
    else
        style{k} = 'S';
    end
end


uq_words = unique(word);

labels = zeros(length(word),1);

for k = 1:length(word)
    labels(k) = find(strcmp(uq_words,word{k}));
end


N_tokens = sum(labels == 1);


if(strcmp(opmode,'single'))

    % Load embeddings

    tmp = dir([embdir '/*.txt']);
    emb_ID = cell(length(tmp),1);
    for k = 1:length(tmp)
        emb_ID{k} = tmp(k).name(1:end-4);
    end

    % Check dimension by loading one
    fid = fopen([embdir '/' tmp(1).name]);
    line = str2num(fgetl(fid));
    fclose(fid);
    dim = length(line);

    X = zeros(length(filenames),dim);

    for k = 1:length(filenames)
        [a,b,c] = fileparts(filenames{k});

        i = find(strcmp(emb_ID,b));
        if(~isempty(i))
            fid = fopen([embdir '/' tmp(i).name]);
            X(k,:) = str2num(fgetl(fid));
            fclose(fid);
        else
            error('cannot find embedding for %s',b);
        end
    end





    % Run a diagnostic classifier

    k_nearest = N_tokens-1;

    recall = zeros(length(uq_words),N_tokens);

    for fold = 1:N_tokens
        i_test = fold:N_tokens:length(labels);
        i_train = setxor(i_test,1:length(labels));

        labels_train = labels(i_train);
        labels_test = labels(i_test);

        D = pdist2(X(i_test,:),X(i_train,:),'cosine');

        [D_sort,D_ind] = sort(D,2,'ascend');

        hypos = labels_train(D_ind(:,1:k_nearest));

        for k = 1:size(hypos,1)
            recall(k,fold) = sum(hypos(k,:) == k)./(N_tokens-1);
        end
    end

    recall = mean(mean(recall,2));

    fprintf('Overall recall: %0.2f%%\n',recall.*100);

elseif(strcmp(opmode,'full'))


    % Load embeddings

    tmp = dir([embdir '/*.txt']);
    emb_ID = cell(length(tmp),1);
    for k = 1:length(tmp)
        emb_ID{k} = tmp(k).name(1:end-4);
    end

    % Check dimension by loading one
    fid = fopen([embdir '/' tmp(1).name]);
    line = str2num(fgetl(fid));
    fclose(fid);
    dim = length(line);

    F = cell(length(filenames),1);

    for k = 1:length(filenames)
        F{k} = zeros(1000,dim);
        [a,b,c] = fileparts(filenames{k});

        i = find(strcmp(emb_ID,b));
        if(~isempty(i))
            c = 1;
            fid = fopen([embdir '/' tmp(i).name]);
            line = 1;
            while(line ~= -1)
                line = fgetl(fid);
                if(line ~= -1)
                    F{k}(c,:) = str2num(line);
                    c = c+1;
                end
            end
            F{k}(c:end,:) = [];
            fclose(fid);
        else
            error('cannot find embedding for %s',b);
        end
    end



    % Version 2: use dtw

    k_nearest = N_tokens-1;

    recall = cell(N_tokens,1);

    D = cell(N_tokens,1);

    if(runparallel)

        parfor fold = 1:N_tokens
            i_test = fold:N_tokens:length(labels);
            i_train = setxor(i_test,1:length(labels));

            labels_train = labels(i_train);
            labels_test = labels(i_test);

            D{fold} = zeros(length(i_test),length(i_train));

            for k = 1:length(i_test)

                Y = F{i_test(k)};
                [row,col] = find(isnan(Y));
                Y(row,:) = [];
                for j = 1:length(i_train)
                    YY = F{i_train(j)};
                    [row,col] = find(isnan(YY));
                    YY(row,:) = [];
                    D{fold}(k,j) = dtw(Y',YY');
                    %[p,q,~,sc] = dpfast(pdist2(Y,YY,'euclidean'));
                    %D{fold}(k,j) = sum(sc);
                end
            end

            [D_sort,D_ind] = sort(D{fold},2,'ascend');

            hypos = labels_train(D_ind(:,1:k_nearest));

            for k = 1:size(hypos,1)
                recall{fold}(k) = sum(hypos(k,:) == k)./(N_tokens-1);
            end
        end

    else
        for fold = 1:N_tokens
            i_test = fold:N_tokens:length(labels);
            i_train = setxor(i_test,1:length(labels));

            labels_train = labels(i_train);
            labels_test = labels(i_test);

            D{fold} = zeros(length(i_test),length(i_train));

            for k = 1:length(i_test)

                Y = F{i_test(k)};
                [row,col] = find(isnan(Y));
                Y(row,:) = [];
                for j = 1:length(i_train)
                    YY = F{i_train(j)};
                    [row,col] = find(isnan(YY));
                    YY(row,:) = [];
                    D{fold}(k,j) = dtw(Y',YY');
                    %[p,q,~,sc] = dpfast(pdist2(Y,YY,'euclidean'));
                    %D{fold}(k,j) = sum(sc);
                end
            end

            [D_sort,D_ind] = sort(D{fold},2,'ascend');

            hypos = labels_train(D_ind(:,1:k_nearest));

            for k = 1:size(hypos,1)
                recall{fold}(k) = sum(hypos(k,:) == k)./(N_tokens-1);
            end
        end
    end

    recall = mean(cellfun(@mean,recall));
    fprintf('Overall recall: %0.2f%%\n',recall.*100);
end





