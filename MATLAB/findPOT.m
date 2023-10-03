function [ excess,peak_ind,lambda,avg_sz,tau ] = findPOT( Hs,srate,windsz,thresh )
%finds independent peaks over a specified threshold value for UNCENSORED Hs data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%When using all time records in a year:
%   Input Hs should be an nx1 vector of time series data of siginificant wave
%   heights. 
%When using a subset of time records in a year (i.e. seasonal data, using
%  Oct to Apr only, etc.:
%   Input Hs should be an nx1 or 1xn cell array of time series data of significant wave
%   heights, where each cell is a vector of time series data from a single season/a single specified
%   time period. For example, if computing Hs return periods from a
%   hindcast ranging from 2000 to 2010 using Oct through Apr data, Hs{1} is
%   the time series from Oct 2000 to Apr 2001, Hs{2} is the time series
%   from Oct 2001 to Apr 2002, etc
%
%Other Input:
%   srate   - The sampling rate, in hours, of the data
%   windsz  - The window size, in hours, to be used to ensure independent
%               peaks
%   thresh  - The threshold, in meters, to be used for determining peaks
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if full set is given, convert to cell array and compute # years in series
if(~iscell(Hs))    
    ny = size(Hs,1)*srate/(24*365.25);  %number of years in time series
    Hs = {Hs};
    n = 1; 
else
    %if cell array is given, the number of years/seasons represented by
        %series is number of cells in array.
    ny = numel(Hs);
    n = ny;
end

%convert window size from 'hours between peaks' to 'observations between peaks'
window = windsz/srate;
excess = [];
peak_ind = [];
tau = 0;
%loop through each season (only one when full dataset is used, jan - dec)
for i = 1:n
    %convert to vector for easier use
    Hs_curr = Hs{i};
    nv = max(size(Hs_curr));
    
    %find all points above the threshold
    ind = find(Hs_curr > thresh);
    
    %avg time between storms
    temp = diff(ind);
    tau = tau + mean(temp(find(temp>windsz/2)));
    
    nex = max(size(ind));
    if(size(ind,1) > 0)
        %identify start and end points of all groups of consecutive points above threshold
        stpt = ind(1);
        endpt = [];
        for i = 2:nex - 1
            if(ind(i + 1) - ind(i) > 1)
                endpt = [endpt;ind(i)];
                stpt = [stpt;ind(i + 1)];            
            end
        end
        endpt = [endpt;ind(end)];
        avg_sz = mean(endpt - stpt);
        %set number clusters
        nclust = size(endpt,1);

        %find maxima for each cluster
        peaks = NaN(nclust,2);
        for i = 1:nclust
            s = stpt(i);
            e = endpt(i);
            [peaks(i,1) tempi] = max(Hs_curr(s:e));
            clustrang = s:e;
            peaks(i,2) = clustrang(tempi);
        end

        %for independence, check that new peak is more than window from old
        %peak. if time between is less than window, use only larger of two. 
        peak_dist = diff(peaks(:,2));
        if(min(size(find(peak_dist<window))) > 0)
            newp = peaks(1,:);
            for i = 2:nclust
                if(peak_dist(i-1) > window)
                    newp = [newp;peaks(i,:)];
                else
                    if(newp(end,1) > peaks(i,1))
                        if(i < nclust)
                            peak_dist(i) = peaks(i+1,2) - newp(end,2);   
                        end
                    else
                        newp(end,:) = peaks(i,:);
                    end
                end
            end

            peaks = newp;
        end

        %compute excesses and index of peaks to return
        excess = [excess;peaks(:,1) - thresh];
        peak_ind = [peak_ind;peaks(:,2)];  
    end
end

%compute average number of clusters per year/season/time frame
lambda = max(size(excess))/ny;
tau = tau/ny;
end

