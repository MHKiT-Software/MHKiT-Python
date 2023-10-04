function [pct, best_thresh] =  threshold(Hs,samp_rate)

    nlags = floor(14*24/samp_rate);
    [acf,lag] = xcorr(Hs - mean(Hs), nlags, 'coeff');
    positive_lag = lag((nlags+1):end);
    positive_acf = acf((nlags+1):end) % CM
    below_thresh = find(acf((nlags+1):end) < 0.5);

    %if it doesn't drop below 0.5 in first 14 days (nlags) then the user should
    %double check their input OR should be fitting this manually to choose the
    %window.
    if isempty(below_thresh)
        fprintf(strcat('ERROR: The acf does not drop below 0.5 in first 14 days, check inputs', ...
                    ' and retry function, or fit manually.\n'))
        HsR = NaN;
        flags = true;
        POT_info = {};
        return
    end

    %set window size (in hours) to the time where acf dropped below 0.5
    windsize = samp_rate*positive_lag(min(below_thresh));

    %set flag if the window is wider than 4 days
    if windsize > 24*4 - 1
        fprintf('WARNING: The acf window is over 4 days. Check final fit.\n')
        flags.large_window_size = true;
    else
        flags.large_window_size = false;
    end

    %save the window size in info structure
    POT_info.window_size = windsize;

    %clear nlags acf lag positive_lag below_thresh

    distribution = 'GeneralizedPareto';
    npar = 3;

    %initialize the first set of thresholds to be tested as the percentiles
    %ranging from 99 to 99.5 percentile
    pct_step = 0.1;
    thresh_pct = [99:pct_step:99.5];
    thresh_test = prctile(Hs,thresh_pct);
    keep_going = true;
    current_best_thresh = -100;
    flags.test_lambda_below_1 = false;

    %loop through until the maximum R is found or the number of peaks per
    %year chosen drops to ~1 per year (annual maxima method)
    while(keep_going)
        thresh_corr = NaN(size(thresh_test));
        for i = 1:length(thresh_test)
            %find peaks over the test threshold
            thresh = thresh_test(i);

            [ excess,peak_ind,lambda,avg_sz ] = findPOT( Hs,samp_rate,windsize,thresh );

            %fit distribution

            POTdist = fitdist(excess,distribution);
            %get qq plot
            QQ_plt = qqplot(excess,POTdist);
            X_data = get(QQ_plt,'Xdata');
            Y_data = get(QQ_plt,'Ydata');
            %find correlation
            thresh_corr(i) = corr(X_data{1}',Y_data{1}');

            %if lambda falls below one, the sample set is now less than one
            %peak per year and should not be used, (might as well use AM)
            if(lambda < 1)
                thresh_corr(i) = 0;
                flags.test_lambda_below_1 = true;
            end

        end
        %find threshold with maximum correlation
        [~,max_i] = max(thresh_corr);

        %if the threshold hasn't changed more than 0.05m OR the number of
        %samples per year has dropped below 2 (~one per year), then quit
        if(abs(current_best_thresh - thresh_test(max_i)) < 0.05 || lambda < 2)
            best_thresh = thresh_test(max_i);
            pct = thresh_pct(max_i);
            keep_going = false;
            if(lambda < 2)
                flags.loop_stopped_by_lambda = true;
            else
                flags.loop_stopped_by_lambda = false;
            end
        %otherwise, find a finer range around the new maximum and loop again
        else
            current_best_thresh = thresh_test(max_i);
            pct_step = pct_step/10;
            if(max_i == length(thresh_test))
                thresh_pct = thresh_pct(max_i-1):pct_step:thresh_pct(max_i)+5*pct_step;
            elseif(max_i == 1)
                thresh_pct = thresh_pct(max_i)-9*pct_step:pct_step:thresh_pct(max_i+1);
            else
                thresh_pct = thresh_pct(max_i-1):pct_step:thresh_pct(max_i+1);
            end
            thresh_test = prctile(Hs,thresh_pct);
        end
    end
end
