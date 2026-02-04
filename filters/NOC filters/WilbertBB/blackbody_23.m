function [TotalPbb,Filtertransmission,NEP,method]=blackbody_23(Tbb,method,plotdata,pad,pOs,ndiv)
% same as 14, but with an additional option for the method.filter: allows
% arbitrary fileread.
% Update 15-12: Error in Bunching term corrected (was ok up to V8))
% updated 6-12-2016 for Deshima filters and added optional pad variable. It
% is then 1:1 the same as the function used in the MUX.
% NEW VERSION 12-12-2015 Jochem. REMOVE STUPID REINIER COLORS
% gets the filter transmisison (from datafiles) and uses this + the throughput, to
% calculate the power as fy of BB Temperature. The sctipt uses
% numerical integartion from the gap frequency to the
% max freq set in the function. Filter transmission is set to 0 outside
% of this range. At the min freq a realistic datapoint is added
% artificially, to get a reasonable estimate *using interplation) of the
% lekage at low F.
%
%INPUT:
% Tbb = A 1xN vector of temperatures of the blackbody.
% method. A struct containing information about the used setup including:
%         filters, lens size, and aperture determination. The following
%         struct elements are required:
% plotdata: 0 or 1 to plot the output figure
% pad = (optional, default=pwd) path where the filterfiles are, can be left out, pad = pwd in that case. it will look in [pad,filesep,'filterfiles',filesep];%
% pOs = (optional, default = 0)  set to 0 to diable any comments on the screen 
% ndiv: (optional, default = 1) devides the number of lines plotted in the plot with ndiv.

%
%   method.pol = Number of polarizations used (equal to either 1 or 2)
%
%   method.filter = '12 THz'            %to be deleivered 12 Tghz (24 um) 2022%
%   method.filter='GBLF'                % Groundbird low frequency (2020)
%   method.filter='8 THz'                % 7.8 THz Superkids stack (2020) - outdated, leaking. %
%   method.filter='7 THz'                % 7 THz Probes stack (2023)
%   method.filter='7 THz +1BP'                % 7 THz Probes stack (2023)  1 extra BP%
%   method.filter='7 THz -1BP'                % 7 THz Probes stack (2023)  1 less BP%
%   method.filter='1_6THz'              % 1.6 THz SPICA SAFARI in separate
%                                       files, or 1_6THz1file for olf method
%   method.filter='350 GHz 4Filters'    %350 Cardiff bandpass with additional LPF
%   method.filter='350 GHz Deshima'     %350 GHz LPF (2x) on cold box, and K1817BPF and K1785 LPG on chip and 1 Thz LPF on BB%
%   method.filter = '650 GHz'           %650 GHz BPF SH (B768), LTbox BPF (B768) and 1 Thz LPF on BB%%
%   method.filter='850 GHz'             %850 GHz filterstack used first with LT010
%   method.filter='FFn'                 %Use a file with filename (full path) 'FFn'. File has that has cols cm^-1 S21 of the filter, tab delimited
%                                       with col 1 in cm^-1 increasing (in frequency)
%                                       "FFn" can be any string, script
%                                       will check if file exists
%
%   method.tp='lambda^2'                %using lambda^2 throughput over the entire filter band%
%   method.tp='Geometrical'             %Using geometrical calculation for throughput determination.
%   
%   ONLY FOR method.tp='Geometrical'%
%   method.lensdiameter=1 %lens diameter in millimeters. 
%   method.opening_angle=10 %angle wrt optical propagation (i.e. total angle is 2x larger) in degrees%
%
%   ONLY FOR method.tp='lambda^2'%
%   method.eta_c = 1.0              %[Optional, Default = 1.00] total SO couplig (CST), only for lambda^2%
%       
%   OPTIONAL    
%   method.GR = 4;                  % prefactor in the GR noise term of the NEP. default = 4
%                                   % following Flanigan et al (APL 108, 083502 (2016). %
%                                   % We used GR=2 in the past (in Pieter/Reineir/SY work%
%   method.freqresolution = 2       %[Optional, Default = 2] frequency resolution in GHz used
%                                       to integrate the blackbody spectrum with all filters.
%                                       NOTE: for the 350 GHz and 325 GHz filters 2 GHz is
%                                       adviced as minimum.
%   method.MaxFreq = 5e12           %[Optional, Default= 5e12] max Freq. of integration%
%   method.Delta = 45.6             %[Optional, Default = 45.6] value of Delta (half
%                                       the superconducting gap) in GHz
%   method.eta_pb = 0.57            %[Optional, Default = 0.57] value of the photon
%                                       pair breaking efficiency. Default from Kozorezov
%                                       et.al. PRB 2000
%
%  plotdata. (either 1 [true] or 0 [false]) If true, the routine plots the
%            photon noise NEP's of Al resonators. 
% 
%OUTPUT: original nput Tbb = 1xN vector
% TotalPbb.             % 1xN vector of blackbody power received by the lens/antenna ateach blackbody temperature%
% method.Etendue.       % Optical throughput of the system at the center frequency of the band %
% method.centrefreq     % central frequency
% method.filterBW       % effective BW
% Filtertransmission.   %2xM vector of total filter transmission [frequency (Hz),transmission].
%                       frequency is given with the resolution specified in method.freqresolution  
% NEP.                  %struct containing four 1xN vectors that for each blackbody temperature give the expected NEP due to:%
%   NEP.g_r         %1xN vector Generation Recombination Noise (using method.Delta) 
%   NEP.poisson     %1xN vector Poissonian photon noise
%   NEP.wave        %1xN vector Wave Bunching Photon Noise
%   NEP.totphoton   %1xN vector Total NEP due to Photon, GR and wavebunching noise.
%
%SUBROUTINES:
% getfilterform (included below)
% plotresults (included below)
%
%REQUIRED FILES (filters, located in /filterfiles subdir):
% S1_5THz.xls
% W1275_350GHz.dat
% W1052 14cm LPE SCUBAII.dat
% B386 18cm LPE.dat
% W969 37cmLPESCUBAII.txt
% K1817_BPFDeshima.txt
% K1785_LPFDeshima.txt
% B768 650GHz BPF.dat
% FP3293 PRIMA 40um BP Jochem.xlsx
% 1941 SRON 38um filters.xlsx
% GB_filter.xls
% PRIMA_25um_Jochem_filters for tests.xlsx


%==========================================================================
% Set overal values and correct for missing optional inputs.
%==========================================================================
warning('off','MATLAB:Axes:NegativeDataInLogAxis');
if nargin == 3
    pad = pwd;
    pOs = 0;
    ndiv = 1; %optional plottig less spectral curves
elseif nargin == 4
    pOs = 0;
    ndiv = 1; %optional plottig less spectral curves
elseif nargin == 5
    ndiv = 1; %optional plottig less spectral curves
end

% Define some global parameters and constants of nature
global c;
h = 6.6262e-34;		% J.s
c = 2.9998e8;		% m/s
k = 1.3806e-23;		% J/K
%==========================================================================
% Set the parameters for the numerical integration. 
%==========================================================================


if isfield(method,'pol') == 0 || isfield(method,'filter') == 0 || isfield(method,'tp') == 0 
    error('method struct not complete');
end

% Check if the polarization was specified correctly
if method.pol~=1 && method.pol~=2
    error('ERROR: Polarization should be either 1 or 2')
end

%check lambda^2 case
if strcmp(method.tp,'lambda^2')
    method.lensdiameter=[];method.opening_angle=[];
    if isfield(method,'eta_c') == 0
        error('No coupling efficiency given.')
    end
end
%check Geometrical case
if strcmp(method.tp,'Geometrical')
    if isfield(method,'lensdiameter') == 0 || isfield(method,'opening_angle') == 0
        error('Geometrical throughput chosen, but no .lensdiameter or .opening_angle defined');
    end
    method.eta_c=1;
    method.solidangle=1*pi*(1-cosd(method.opening_angle)^2); %including  effective area reduction at large angles
end
%==========================================================================
% Check if the optional parameters are given as input.
%==========================================================================
if isfield(method,'MaxFreq') == 0
    method.MaxFreq = 10e12;
end
if isfield(method,'GR') == 0
    method.GR = 4;
end
if isfield(method,'Delta') == 0
    method.Delta = 45.6;     % gap estimate Aluminium in GHz, from H10
end
delta = method.Delta*1e9*h; % gap value in J

% Pair breaking efficiency
if isfield(method,'eta_pb') == 0
   method.eta_pb = 0.57;   % 0.57 pair breaking efficiency by Kozorezov et.al. RPB 2000
   %went to the value of 0.4 using the results from Guruswamy et al. (2015)
end

% Frequency Resolution used for integration
if isfield(method,'freqresolution') == 0
    method.freqresolution = 2;     % Frequency resolution default of 2 GHz
end

%correct method for throughput
if isfield(method,'tp') == 0 || ~(strcmp(method.tp,'lambda^2') || strcmp(method.tp,'Geometrical'))
    error('No thropughput defined')     % Perfect alignment
end

% Show the specified method as a double check for the user.
if pOs ~=0
    fprintf('Specified Method for Blackbody Power calculation:\n')
    disp(method);
end
%==========================================================================
% Start script
%==========================================================================

% Set the parameters for the numerical integration. 
minfrequency = 2*method.Delta*1e9; %minimum frequency [Hz]: gap frequency in Hz.
freq = minfrequency:(method.freqresolution*1e9):method.MaxFreq;
nowave = length(freq);

% Get the filters and interpolate of obtain the values at the desired frequencies.%
[filterform, legendstrr]=getfilterform(method,pad,pOs);% filterform is a cell array with the raw filter specs

%Interpolate the data from file to obtain transmission at the desired integration frequencies.%
filter = ones(1,nowave);        %start the overall filter with full transmission
for I=1:nowave                  %loop over each frequency
    for n=1:length(filterform)  %loop over all consecutive filters
        if freq(I)/c > max(filterform{n}(:,1)) || freq(I)/c < min(filterform{n}(:,1))
            filter(I) = 0;% Outside the filter range. No transmission is assumed.
        else
            % Inside the filter range. Use interpolation to obtain value at
            % specified frequency. Multiply the current transmission with
            % the interpolation value.
            filter(I)=filter(I) * interp1(filterform{n}(:,1),filterform{n}(:,2),freq(I)/c,'pchip','extrap');
        end
    end
end
%Prepare the filter transmission for output.
filter((filter<0))=0;
Filtertransmission = [freq',filter']';
method.centrefreq=sum(freq.*filter)/sum(filter);                    % central frequency
method.filterBW=sum(filter*(method.freqresolution*1e9))/max(filter); % effective BW
%==========================================================================
% Numerical Integration of the Planck function over all frequencies for all
% temperatures requested.
%==========================================================================
% initialize arrays
N_Tbb=length(Tbb); %number of blackbody temperatures
irradiation=cell(1,length(Tbb));powerbla2=cell(1,length(Tbb));filterirad2=cell(1,length(Tbb));
TotalPbb=zeros(1,N_Tbb); %Total power arriving at lens/antenna
NEP.g_r=zeros(1,N_Tbb); %GR noise
NEP.poisson=zeros(1,N_Tbb); %Poissonian photon noise
NEP.wave=zeros(1,N_Tbb); %Wavebunching photon noise
NEP.totphoton=zeros(1,N_Tbb); %Total noise due to photon fluctuations

%Numerical integration of Planck spectral radiance for all blackbody temperatures
%the hroughput is included here for each individual frequency
for p=1:length(Tbb)
    %Photon Occupation Number (Bose-Einstein distribution):
    occupation=1./(exp(h*freq/(k*Tbb(p)))-1);
    %Planck brillance W/(m^2 str Hz) [B_{\nu}(T_{bb})]:
    brilliance=(method.pol*h*freq).* (freq.^2/(c^2)) .*occupation;%also called brightness
    % total irradiation with fixed throughput but without filters in W/Hz
    % [B_{\nu}(T_{bb})*A\Omega]:
    if strcmp(method.tp,'lambda^2')
            Etendue=(c./freq).^2; %array
            method.Etendue=(c/method.centrefreq)^2;
        elseif strcmp(method.tp,'Geometrical')
            Etendue=method.solidangle*pi*(method.lensdiameter/2)^2+zeros(1,length(freq));
            method.Etendue=method.solidangle*pi*(method.lensdiameter/2)^2;
        else
            error('no correct string for thoughput option');
    end
    irradiation{p}=brilliance.*Etendue; 
    % total irradiation @ lens front in W/Hz
    % [B_{\nu}(T_{bb})*A\Omega*F_{\nu}]:
    filterirad2{p}=irradiation{p}.*filter.*method.eta_c;
    % Power per frequency bin in W.
    % [B_{\nu}(T_{bb})*A\Omega*F_{\nu} d\nu]:
    powerbla2{p}=filterirad2{p}*method.freqresolution*1e9;
    % Total received power [W] Summation over power in all F bins
    TotalPbb(p)=sum(powerbla2{p});
    
    % Calculation of the various NEP's
    NEP.g_r(p)=sqrt(sum(method.GR*powerbla2{p}*delta/method.eta_pb)); %g-r noise, only recombination
    NEP.poisson(p)=sqrt(sum(2*powerbla2{p}.*h.*freq)); %Poisson noise term
    
    %lambda2=(c./freq).^2; %lamdba^2
    % wave bunching: 2Phf        * opt coupling source - detector * occupation%
    wave=(2*powerbla2{p}.*h.*freq).*(filter.*method.eta_c).*occupation; 
    NEP.wave(p)=sqrt(sum(wave)); %wave contribution
end
NEP.totphoton=sqrt(NEP.g_r.^2+NEP.poisson.^2+NEP.wave.^2); %total KID NEP

%==========================================================================
% Plotting and warp up.
%==========================================================================
if plotdata==1
    %Plot the total power, filter transmission and NEPs if desired.
    plotresults(Tbb,NEP,filter,TotalPbb,filterform,freq,filterirad2,legendstrr,method,ndiv,irradiation);
end

%END OF MAIN PROGRAM
end

function plotresults(Tbb,NEP,filter,powersum,filterform,freq,filterirad2,legendstrr, method, ndiv,irradiation)
%This plotting function shows the results of the blackbody main
%program. It requires 6 variables from this function as input.
%creates one figure
global c;

%FIGURE 
figure(123451);
%clf %clears figure window


%filter T
subplot(3,2,1)
kolors = colormapJetJBBB(length(filterform));
semilogy(freq(:)/1e12,filter(:),'color','k','LineWidth',3);hold on
for Nfilters=1:length(filterform)
    semilogy(c.*filterform{Nfilters}(:,1)/1e12,filterform{Nfilters}(:,2),'color',kolors(Nfilters,:),'LineWidth',2)
end
ylabel('Transmission')
xlabel('Frequency (THz)')
axis([0 ceil(2*method.centrefreq/1e12) 0.99e-5 1]);
legend([{'all'},legendstrr],'Location','Best');
yticks([10^-5 10^-4 10^-3 10^-2 10^-1 10^0]);
hold off
grid on

subplot(3,2,2)%plot xlim is limited by axis tight in log sacle to values > 0
loglog(freq(:)/1e12,filter(:),'k-','linewidth',2);hold on
loglog([method.centrefreq-method.filterBW/2 method.centrefreq-method.filterBW/2 ...
    method.centrefreq+method.filterBW/2 method.centrefreq+method.filterBW/2]/1e12,...
    [1e-15 max(filter) max(filter) 1e-15],'-b')
loglog([method.centrefreq method.centrefreq]/1e12,...
    [1e-15  max(filter)],'-r')
%xlim([0.09 method.MaxFreq/1e12])
%ylim([1e-15 1.2*max(filter)])
xlabel('Frequency (THz)')
ylabel('Filter Transmission')
grid on
title(['\nu_0 = ' num2str(round(method.centrefreq/1e9)) ' BW = ' num2str(round(method.filterBW/1e9)) ' GHz'])
hold off

%Overview of the power received at each wavelength
subplot(3,2,3)
Tsweep = colormapJetJBBB( ceil(length(Tbb)/ndiv) ) ;
nmn=1;
for Nbb=1:ndiv:length(Tbb)
    semilogy(freq(:)/1e12,filterirad2{1,Nbb}(:),'color',Tsweep(nmn,:),'LineWidth',2);hold on
    semilogy(freq/1e12,irradiation{1,Nbb}(:),'-','color',Tsweep(nmn,:),'LineWidth',1);
    legend('Filtered Irradiation','Irradiation','autoupdate','off');
    plottedTBB(nmn)=Tbb(Nbb);
    plottedTBBtext{nmn}=num2str(Tbb(Nbb),'%.0f');
    nmn=nmn+1;
end
grid on
ylabel('I_\nu(T_{BB}) \lambda^2 F(\nu) ')
xlabel('Frequency (THz)')
axis tight
ylim([1e-50 1e-20] )
hold off
colormap(Tsweep);
% h=colorbar('Ticks',[0, 1],...
%          'TickLabels',{plottedTBB(1) , plottedTBB(end)});
h=colorbar;
h.YTick = ((1:length(plottedTBB))-0.5)/length(plottedTBB);
h.YTickLabel = plottedTBBtext;
h.Label.String='T_{BB}[K]';

%Overview of the power received at each wavelength
subplot(3,2,4)
nmn=1;
nmn=1;
for Nbb=1:ndiv:length(Tbb)
    plot(freq(:)/1e12,filterirad2{1,Nbb}(:)/max(filterirad2{1,Nbb}(:)),'color',Tsweep(nmn,:),'LineWidth',2);hold on
    plottedTBB(nmn)=Tbb(Nbb);
    plottedTBBtext{nmn}=num2str(Tbb(Nbb),'%.0f');
    nmn=nmn+1;
end
grid on
ylabel('normalized I_s(\nu,T_{BB}) \lambda^2 F(\nu) ')
xlabel('Frequency (THz)')
axis tight
% ylim([0 1]*max(powerbla2{1,end}(:)*1.05))
ylim([0 1]*1.05);
hold off
colormap(Tsweep);



%Total received power [W] as a function of Tbb
subplot(3,2,5)
loglog(Tbb,powersum,'LineWidth',2)
grid on
hold on
legend('Integrated power','Location','northwest')
ylabel('Recieved Power [W]')
xlabel('T_{BB} [K]')

%Overview of the calculated NEP's use to photon induced pair breaking
subplot(3,2,6)
loglog(Tbb,NEP.poisson,'LineWidth',2)
hold on
loglog(Tbb,NEP.wave,'g','LineWidth',2)
loglog(Tbb,NEP.g_r,'r','LineWidth',2)
loglog(Tbb,NEP.totphoton,'k','LineWidth',2)
legend('Poisson NEP','wave NEP','g-r NEP','NEPtot','Location','Best')
ylabel('NEP_{Blip} [W/\surd{Hz}]')
xlabel('T_{BB} [K]')
grid on


%END OF plotresults
end

function [filterform, legendstrr, oldfilterform]=getfilterform(method,pad,pOs)
% This function uses the "method.filter" component of the "method" struct to
% determine the used filters. It then loads their filter characteristics
% from file. And returns the cell-array "filterform" of length N. Here N is
% the number of filters in the setup. Each cell contains an 2xM_i array with
% [1/lambda filtertransmission], with 1/lambda decreasing (= F decreasing). NB: at the lowest F point (= last point) a single
% datapoint is patched to all filters: 0 for BPF/HPF and 1 for LPF. In the
% integration interplation will be used between the last real datapoint and
% this point.

% 

%Determine path to the filterfiles.
FilterPath = [pad,filesep,'filterfiles',filesep];
if pOs ~= 0
    disp(['BlackBody: Looking for filter files in: '])
    disp([FilterPath]);
end

global c;
% Minimum frequency in 1/lambda m^-1 to patch some filters.


if strcmp(method.filter,'1_6THz') % 1.6 THz SPICA SAFARI stack) 
    % 1.55 THz individual filters being read in 
    % Filter data read
    xldat       = readmatrix([FilterPath 'S1_5THz.xls'],'Sheet','W1256','Range','A1:B993');% 'FileType','spreadsheet'
    W1256(:,1)  = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    W1256(:,2) 	= xldat(:,2); 
    %add data at high F
    al = length(W1256);
    W1256(al+1,1)  = 6e4; %same as other files
    W1256(al+1,2)  = W1256(al,2);
    clear xldat al;    
    xldat       = readmatrix([FilterPath 'S1_5THz.xls'],'Sheet','W1111','Range','A1:B2302');% 
    W1111(:,1)  = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    W1111(:,2) 	= xldat(:,2); 
    clear xldat ;    
    xldat       = readmatrix([FilterPath 'S1_5THz.xls'],'Sheet','B746','Range','A1:B2302');% 
    B746(:,1)  = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    B746(:,2) 	= xldat(:,2); 
    clear xldat ;    
    xldat       = readmatrix([FilterPath 'S1_5THz.xls'],'Sheet','W933','Range','A1:B2302');% 
    W933(:,1)  = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    W933(:,2) 	= xldat(:,2); 
    clear xldat ;    
    xldat       = readmatrix([FilterPath 'S1_5THz.xls'],'Sheet','B724','Range','A1:B2284');% 
    B724(:,1)  = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    B724(:,2) 	= xldat(:,2); 
    clear xldat ; 
	xldat       = readmatrix([FilterPath 'S1_5THz.xls'],'Sheet','W1467','Range','A1:B2306');% 
    W1467(:,1)  = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    W1467(:,2) 	= xldat(:,2); 
    clear xldat ;    
  
    %Radiator: W1256 - B746 - W933
    filterform{1}(:,1)=[0 W1256(:,1)' ];                %adding low F point  
    filterform{1}(:,2)=[W1256(1,2) W1256(:,2)' ];   
    filterform{2}(:,1)=[0 B746(:,1)' ];                %adding low F point  
    filterform{2}(:,2)=[B746(1,2) B746(:,2)' ];   
    filterform{3}(:,1)=[0 W933(:,1)' ];                %adding low F point  
    filterform{3}(:,2)=[W933(1,2) W933(:,2)' ]; 
    %100 mK: W1111 - B724
    filterform{4}(:,1)=[0 W1111(:,1)' ];                %adding low F point  
    filterform{4}(:,2)=[W1111(1,2) W1111(:,2)' ];   
    filterform{5}(:,1)=[0 B724(:,1)' ];                %adding low F point  
    filterform{5}(:,2)=[B724(1,2) B724(:,2)' ];   
    % chip: W1256 - B746 - W1467
    filterform{6}(:,1)=[0 W1256(:,1)' ];                %adding low F point  
    filterform{6}(:,2)=[W1256(1,2) W1256(:,2)' ];   
    filterform{7}(:,1)=[0 B746(:,1)' ];                %adding low F point  
    filterform{7}(:,2)=[B746(1,2) B746(:,2)' ];   
    filterform{8}(:,1)=[0 W1467(:,1)' ];                %adding low F point  
    filterform{8}(:,2)=[W1467(1,2) W1467(:,2)' ]; 

    legendstrr={'BB: W1256','BB: B746','BB: W933',...
        'LTbox: W1111','LTbox: B724',...
        'Holder: W1256','Holder: B746','Holder: W1467'};
elseif strcmp(method.filter,'1_6THz1file') %1.55 THz in 2 file
    % Added JB/PdV 21_9_2012 - reading 1 file for all in 1.55 THz
    bba=flipdim(dlmread([FilterPath,'totalLbandfilters.txt'],'\t'),1);
    filterform{1}(:,1)=[bba(:,1)'*100 0];    % from cm-1 to m-1
    filterform{1}(:,2)=[bba(:,2)' 0];        % include optical transmission estimate EP
    legendstrr='All filters combined';
    %SAFARI_test_filters_June09_dataonly

elseif strcmp(method.filter,'350 GHz 4Filters') %350 GHz BP from cardiff with additional LPF
    %BPF holder
    bba=flipdim(dlmread([FilterPath,'W1275_350GHz.dat'],'\t'),1);%BPF Sample Holder
    filterform{1}(:,1)=[bba(:,1)'*100 ];  % from cm-1 to m-1
    filterform{1}(:,2)=[bba(:,2)' ];%not patching 0 transmission at lowest F as this is a full BPF stack, 0 is already in file
    %LPFs on cold box
    bba=flipdim(dlmread([FilterPath,'W1052 14cm LPE SCUBAII.dat'],'\t'),1);
    filterform{2}(:,1)=[bba(:,1)'*100 0];
    filterform{2}(:,2)=[bba(:,2)' 1];%
    bba=flipdim(dlmread([FilterPath,'B386 18cm LPE.dat'],'\t'),1);
    filterform{3}(:,1)=[bba(:,1)'*100 0];
    filterform{3}(:,2)=[bba(:,2)' 1];%
    %4K LPF
    bba=flipdim(dlmread([FilterPath,'W969 37cmLPESCUBAII.txt'],'\t'),1);%KLPF BB
    filterform{4}(:,1)=[bba(:,1)'*100 0];
    filterform{4}(:,2)=[bba(:,2)' 1];%LPF, add 1 out of range at low F
    legendstrr={'W1275 BPF holder','W1052 LPF box','B386 LPF box','W696 LPF BB'};

elseif strcmp(method.filter,'350 GHz Deshima') %350 GHz BP from cardiff with additional LPF
    %BPF+LPF holder
    bba=flipdim(dlmread([FilterPath,'K1817_BPFDeshima.txt'],'\t'),1);%BPF Sample Holder
    filterform{1}(:,1)=[bba(:,1)'*100 0];  % from cm-1 to m-1
    filterform{1}(:,2)=[bba(:,2)' 0];%patcghing 0 transmission at lowest F as this is a full BPF stack
    bba=flipdim(dlmread([FilterPath,'K1785_LPFDeshima.txt'],'\t'),1);%BPF Sample Holder
    filterform{2}(:,1)=[bba(:,1)'*100 0];  % from cm-1 to m-1
    filterform{2}(:,2)=[bba(:,2)' 0];%patcghing 0 transmission at lowest F as this is a full BPF stack
    %LPFs on cold box
    bba=flipdim(dlmread([FilterPath,'W1052 14cm LPE SCUBAII.dat'],'\t'),1);
    filterform{3}(:,1)=[bba(:,1)'*100 0];
    filterform{3}(:,2)=[bba(:,2)' 1];%LPF, add 1 out of range at low F
    bba=flipdim(dlmread([FilterPath,'B386 18cm LPE.dat'],'\t'),1);
    filterform{4}(:,1)=[bba(:,1)'*100 0];
    filterform{4}(:,2)=[bba(:,2)' 1];%LPF, add 1 out of range at low F
    %4K LPF
    bba=flipdim(dlmread([FilterPath,'W969 37cmLPESCUBAII.txt'],'\t'),1);%KLPF BB
    filterform{5}(:,1)=[bba(:,1)'*100 0];
    filterform{5}(:,2)=[bba(:,2)' 1];%LPF, add 1 out of range at low F
    legendstrr={'K1817 BPF holder','K1785 LPF holder','W1052 LPF box','B386 LPF box','W696 LPF BB',};
    
elseif strcmp(method.filter,'850 GHz') %850 GHz BP from cardiff with additional LPF
    %Added RJ 2013/12/16
    K1979=flipdim(dlmread([FilterPath,'K1979 855GHz BPF.dat'],'\t'),1);
    K1981=flipdim(dlmread([FilterPath,'K1981 1140GHz LPF.dat'],'\t'),1);
    K1980=flipdim(dlmread([FilterPath,'K1980 990GHz LPF.dat'],'\t'),1);
    %B624=flipdim(dlmread([FilterPath,'B624 660GHz HPF.dat'],'\t'),1); We
    %do not have rthis filter, was here for hystorical reasons and is very
    %similar to B588
    B588=flipdim(dlmread([FilterPath,'B588 HPF.dat'],'\t'),1);
    
    %Filters on the 4K blackbody
    filterform{1}(:,1)=[K1980(:,1)'*100 0];  % Add 1 datapoint at lowest F
    filterform{1}(:,2)=[K1980(:,2)' 1];%LPF, add 1 out of range at low F
    filterform{2}(:,1)=[B588(:,1)'*100 0];
    filterform{2}(:,2)=[B588(:,2)' 0];%HPF, add 0 out of range at low F
    filterform{3}(:,1)=[K1979(:,1)'*100 0];
    filterform{3}(:,2)=[K1979(:,2)' 0];%BPF, add 0 out of range at low F
    
    %Filters on the 100mK outer box
    filterform{4}(:,1)=[K1981(:,1)'*100 0];  % from cm-1 to m-1
    filterform{4}(:,2)=[K1981(:,2)' 1];%LPF, add 1 out of range at low F
    filterform{5}(:,1)=[K1979(:,1)'*100 0]; 
    filterform{5}(:,2)=[K1979(:,2)' 0];%BPF, add 0 out of range at low F
    
    %Filters on the 100mK sample
    filterform{6}(:,1)=[K1980(:,1)'*100 0];  % from cm-1 to m-1
    filterform{6}(:,2)=[K1980(:,2)' 1];%LPF, add 1 out of range at low F
    filterform{7}(:,1)=[B588(:,1)'*100 0];
    filterform{7}(:,2)=[B588(:,2)' 0];%HPF, add 0 out of range at low F
    filterform{8}(:,1)=[K1979(:,1)'*100 0];
    filterform{8}(:,2)=[K1979(:,2)' 0];%BPF, add 0 out of range at low F
    
    legendstrr={'K1980 LPF BB','B588 HPF BB','K1979 BPF BB','K1981 LPF Box','K1979 BPF Box','K1980 LPF holder','B588 HPF holder','K1979 BPF holder'};

elseif strcmp(method.filter,'650 GHz') %650 GHz BP from cardiff 
    %Added JB 2017/04/05
    B768=flipdim(dlmread([FilterPath,'B768 650GHz BPF.dat'],'\t'),1); 
    K2136 = flipdim(dlmread([FilterPath,'K2136_450um_bp.txt'],'\t'),1); 
    
    %4K LPF (from 350 ghz setup)
    bba=flipdim(dlmread([FilterPath,'W969 37cmLPESCUBAII.txt'],'\t'),1);%KLPF BB
    filterform{1}(:,1)=[bba(:,1)'*100 0];
    filterform{1}(:,2)=[bba(:,2)' 0];%LPF, add 1 out of range at low F
    
    %Filter on the LT box
    filterform{2}(:,1)=[B768(:,1)'*100 0];  % from cm-1 to m-1
    filterform{2}(:,2)=[B768(:,2)' 0];%BPF, add 1 out of range at low F
    
    %Filter on the SH
    filterform{3}(:,1)=[B768(:,1)'*100 0];  % from cm-1 to m-1
    filterform{3}(:,2)=[B768(:,2)' 0];%BPF, add 1 out of range at low F
    legendstrr={'W969 LPF BB','K2136 650GHz BPF','B768 650GHz BPF'};

elseif strcmp(method.filter,'7 THz')
%     1: upgrade 7.8 THz stack
%       Four new inductive BPF ~240 cm-1: FP3293
%           2x 25.4 mm diameter 
%           1x 27.5 mm diameterˀ
%           1x 11.5 mm diameter for the new He7 cooler
% Two HPF W3045 200 cm-1 HPE
%           2x 25.4 mm diameter
    % Filter data read
    xldat   = readmatrix([FilterPath, 'FP3293 PRIMA 40um BP Jochem'],'Sheet','C0298_17','Range','A2:B12181');%FP3293 BP, 1x27.5 mounted, 2x 25.4, 1 mounted
    m_1     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    BPF     = xldat(:,2); %2 BPF 
    clear xldat ;    
    
    xldat   = readmatrix([FilterPath, 'FP3293 PRIMA 40um BP Jochem'],'Sheet','T1896R7','Range','A2:B2368');% W3045 6THz HPF. 3 total, 2 25.4, 1 27.5. data  has same cm-1 axis as BP filter - not needed to read again
    m_2     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPF     = abs(xldat(:,2)); %6 THz HPF
    clear xldat
    % 
    %Radiator: BPF FP3293 %HPF W3045 
    filterform{1}(:,1)=[0 m_2(200:end)' m_1(end)];       %Low F region: set to 1e-3 for the first 100 datapoints (info carole Tucker). High point added as well%
    filterform{1}(:,2)=abs([1e-3 HPF(200:end)' HPF(end)]); %

    %LT box snout HPF W3045 
    filterform{2}(:,1)=[0 m_2(200:end)' m_1(end)];       %Low F region: set to 1e-3 for the first 100 datapoints (info carole Tucker). High point added as well%
    filterform{2}(:,2)=abs([1e-3 HPF(200:end)' HPF(end)]); %

    %LT box BPF FP3293 
    filterform{3}(:,1)=[0 m_1' ];       %adding low F point at 0 frequency
    filterform{3}(:,2)=abs([BPF(1) BPF' ]);  %adding low F point = lowest meaured datapoint 
    
    %sample BPF FP3293 HPF W3045 
    filterform{4}(:,1)=[0 m_1' ];       %adding low F point at 0 frequency
    filterform{4}(:,2)=abs([BPF(1) BPF' ]);  %adding low F point = lowest meaured datapoint 

    legendstrr={'BB: W3045 6THz HPF','LTbox snout: W3045 6THz HPF','LTbox: FP3293 7THz BPF','Sample: FP3293 7THz BPF'};
    if method.MaxFreq < 9e12
        error('Max Frequency in integration < 9 Thz for 8 Thz filter stack');
    end

elseif strcmp(method.filter,'7 THz +1BP')
    % Filter data read
    xldat   = readmatrix([FilterPath, 'FP3293 PRIMA 40um BP Jochem'],'Sheet','C0298_17','Range','A2:B12181');%FP3293 BP, 1x27.5 mounted, 2x 25.4, 1 mounted
    m_1     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    BPF     = xldat(:,2); %2 BPF 
    clear xldat ;    
    
    xldat   = readmatrix([FilterPath, 'FP3293 PRIMA 40um BP Jochem'],'Sheet','T1896R7','Range','A2:B2368');% W3045 6THz HPF. 3 total, 2 25.4, 1 27.5. data  has same cm-1 axis as BP filter - not needed to read again
    m_2     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPF     = abs(xldat(:,2)); %6 THz HPF
    clear xldat
    % 
    %Radiator: BPF FP3293 %HPF W3045 
    filterform{1}(:,1)=[0 m_2(200:end)' m_1(end)];       %Low F region: set to 1e-3 for the first 100 datapoints (info carole Tucker). High point added as well%
    filterform{1}(:,2)=abs([1e-3 HPF(200:end)' HPF(end)]); %

    %LT box snout HPF W3045 + BPF FP3293
    filterform{2}(:,1)=[0 m_2(200:end)' m_1(end)];       %Low F region: set to 1e-3 for the first 100 datapoints (info carole Tucker). High point added as well%
    filterform{2}(:,2)=abs([1e-3 HPF(200:end)' HPF(end)]); %
    filterform{3}(:,1)=[0 m_1' ];       %adding low F point at 0 frequency
    filterform{3}(:,2)=abs([BPF(1) BPF' ]);  %adding low F point = lowest meaured datapoint 

    %LT box BPF FP3293 
    filterform{4}(:,1)=[0 m_1' ];       %adding low F point at 0 frequency
    filterform{4}(:,2)=abs([BPF(1) BPF' ]);  %adding low F point = lowest meaured datapoint 
    
    %sample BPF FP3293 HPF W3045 
    filterform{5}(:,1)=[0 m_1' ];       %adding low F point at 0 frequency
    filterform{5}(:,2)=abs([BPF(1) BPF' ]);  %adding low F point = lowest meaured datapoint 

    legendstrr={'BB: W3045 6THz HPF','LTbox snout: W3045 6THz HPF','LTbox snout: FP3293 7THz BPF','LTbox: FP3293 7THz BPF','Sample: FP3293 7THz BPF'};
    if method.MaxFreq < 9e12
        error('Max Frequency in integration < 9 Thz for 8 Thz filter stack');
    end

elseif strcmp(method.filter,'7 THz -1BP')
    % Filter data read
    xldat   = readmatrix([FilterPath, 'FP3293 PRIMA 40um BP Jochem'],'Sheet','C0298_17','Range','A2:B12181');%FP3293 BP, 1x27.5 mounted, 2x 25.4, 1 mounted
    m_1     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    BPF     = xldat(:,2); %2 BPF 
    clear xldat ;    
    
    xldat   = readmatrix([FilterPath, 'FP3293 PRIMA 40um BP Jochem'],'Sheet','T1896R7','Range','A2:B2368');% W3045 6THz HPF. 3 total, 2 25.4, 1 27.5. data  has same cm-1 axis as BP filter - not needed to read again
    m_2     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPF     = abs(xldat(:,2)); %6 THz HPF
    clear xldat
    % 
   %Radiator: BPF FP3293 %HPF W3045 
    filterform{1}(:,1)=[0 m_2(200:end)' m_1(end)];       %Low F region: set to 1e-3 for the first 100 datapoints (info carole Tucker). High point added as well%
    filterform{1}(:,2)=abs([1e-3 HPF(200:end)' HPF(end)]); %

    %LT box snout HPF W3045 
    filterform{2}(:,1)=[0 m_2(200:end)' m_1(end)];       %Low F region: set to 1e-3 for the first 100 datapoints (info carole Tucker). High point added as well%
    filterform{2}(:,2)=abs([1e-3 HPF(200:end)' HPF(end)]); %

    %sample BPF FP3293 HPF W3045 
    filterform{3}(:,1)=[0 m_1' ];       %adding low F point at 0 frequency
    filterform{3}(:,2)=abs([BPF(1) BPF' ]);  %adding low F point = lowest meaured datapoint 

    legendstrr={'BB: W3045 6THz HPF','LTbox snout: W3045 6THz HPF','Sample: FP3293 7THz BPF'};
    if method.MaxFreq < 9e12
        error('Max Frequency in integration < 9 Thz for 8 Thz filter stack');
    end

elseif strcmp(method.filter,'8 THz')
    % Filter data read
    xldat   = readmatrix([FilterPath, '1941 SRON 38um filters'],'Sheet','K2993','Range','A2:D2604');%band pass
    m_1     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    BPF     = xldat(:,2); %2 BPF 
    clear xldat ;    
    xldat   = readmatrix([FilterPath, '1941 SRON 38um filters'],'Sheet','K2994','Range','A2:D2604');%has same cm-1 axis as BP filter - not needed to read again
    HPF     = abs(xldat(:,2)); %6 THz HPF
    clear xldat 
    xldat   = readmatrix([FilterPath, 'SRON Spacekids 1.5THz data Sept2015.xlsx'],'Sheet','K2328','Range','A2:B3032');%
    m_2     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPF2    = abs(xldat(:,2)); %1 THz HPF
    clear xldat
    % 
    %Radiator: BPF K2993 + HPF K2994 + HPF2 K2328
    filterform{1}(:,1)=[0 m_1' ];       %adding low F point at 0 frequency
    filterform{1}(:,2)=[BPF(1) BPF' ];  %adding low F point = lowest meaured datapoint 
    filterform{2}(:,1)=[200 m_1' ];     %HPF 200cm-1: adding low F point based upon data Steohen: 60 Ghz = 2cm-1=200m-1 has T 0.1
    filterform{2}(:,2)=[0.1 HPF']; %
    filterform{3}(:,1)=[0 m_2' ];       %Safari old HPF: use lowest measured datapoint as T at 0 m-1
    filterform{3}(:,2)=[HPF2(1) HPF2']; %
    %LT box: 2x  all these 3: BPF K2993 + HPF K2994 + HPF2 K2328
    filterform{4}(:,1)=[0 m_1' ]; %adding low F point at 0
    filterform{4}(:,2)=[BPF(1) BPF' ]; %adding low F point 
    filterform{5}(:,1)=[0 m_1' ]; %adding low F point at 0
    filterform{5}(:,2)=[BPF(1) BPF' ]; %adding low F point 
    filterform{6}(:,1)=[200  m_1' ]; 
    filterform{6}(:,2)=[0.1 HPF']; %
    filterform{7}(:,1)=[200  m_1' ]; 
    filterform{7}(:,2)=[0.1 HPF']; %
    filterform{8}(:,1)=[0 m_2' ]; 
    filterform{8}(:,2)=[HPF2(1) HPF2']; %
    %Sample: BPF K2993 and LPF K2994
    filterform{9}(:,1)=[0 m_1' ]; 
    filterform{9}(:,2)=[BPF(1) BPF']; 
    filterform{10}(:,1)=[200 m_1' ]; 
    filterform{10}(:,2)=[0.1 HPF']; %

    legendstrr={'BB: K2993 8THz BPF','BB: K2994 6THz HPF','BB: K2328 1THz HPF',...
        'LTB: K2993 8THz BPF','LTB: K2993 8THz BPF','LTB: K2994 6THz HPF','LTB: K2994 6THz HPF','LTB: K2328 1THz HPF',...
        'Chip: K2993 8THz BPF','Chip: K2994 6THz HPF'};
    if method.MaxFreq < 9e12
        error('Max Frequency in integartion < 9 Thz for 8 Thz filter stack');
    end
elseif strcmp(method.filter,'GBLF')%groundird: 2 filters (K1890 and K1900) on holder, none on LT box, BB 1 Thz LPF
    % Filter data read Groundbird specific
    % K1890 HPF
    xldat   = readmatrix([FilterPath,'GB_filter'],'Sheet','T1625R7','Range','A2:B359');%K1890 HPF
    m_1     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPF     = xldat(:,2); %
    filterform{1}(:,1)=[0 m_1' ]; %adding low F point of 0
    filterform{1}(:,2)=[1e-5 HPF']; %adding low F point of 0
    clear xldat m_1;    
    %K1900 LPF
    xldat   = readmatrix([FilterPath,'GB_filter'],'Sheet','T1627R22','Range','A2:B379');%K1900 LPF 
    m_1     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    LPF     = abs(xldat(:,2)); %2 (no LPF), 3 (Combined LPF+BPF)
    filterform{2}(:,1)=[0 m_1' ]; %adding low F point of 0
    filterform{2}(:,2)=[1 LPF']; %adding low F point of 0

    %4K LPF from 350 GHz setup
    bba=flipdim(dlmread([FilterPath,'W969 37cmLPESCUBAII.txt'],'\t'),1);%KLPF BB
    filterform{3}(:,1)=[bba(:,1)'*100 0];
    filterform{3}(:,2)=[bba(:,2)' 1];%LPF, add 1 out of range at low F - data from high to low F
    legendstrr={'K1890 HPF holder','K1900 LPF holder','W696 LPF BB',};

elseif strcmp(method.filter,'12 THz')
    lowT_outband = 1e-3; % data for low F transmission below 100 cm-1 for HPE. Following disciusions with Peter Ade and Carole Tucker, imn agreement with data at low F that is available
    % no patching at high F needed - all data up to 5000 cm-1 or more
    % Filter data read
    % BAND PASS = 1
    xldat   = readmatrix([FilterPath, 'PRIMA_25um_Jochem_filters for tests'],'Sheet','C0285_4','Range','A2:B12181');%FP3223 25um band pass, 2x 25.4, 1x 27.7
    m_1     = 100*xldat(:,1); %convert to m-1, data is in cm-1  %
    BPF     = abs(xldat(:,2)); %2 BPF 
    clear xldat ;   
    % LOW PASS = 2
    xldat   = readmatrix([FilterPath, 'PRIMA_25um_Jochem_filters for tests'],'Sheet','C0288_6','Range','A2:B10124');%FP3253 600cm-1 LPE, 1x 25.4, 1x 27.5
    m_2     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    LPF     = abs(xldat(:,2)); %6 THz HPF
    clear xldat
    % HIGH PASSES = 3(chip) 4(?) 5(rest). for F < 100 cm-1 transmission 1e-2 is used. The measured values are FTS leakage acoridmng to Peter Ade and Carole Tucker from QMCI/Cardiff.
    xldat   = readmatrix([FilterPath, 'PRIMA_25um_Jochem_filters for tests'],'Sheet','C0290_5','Range','A2:B12181');%FP3251 300 cm-1HPE, 1x 27.5
    m_3     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPFchip = abs(xldat(:,2)); %6 THz HPF
    clear xldat
    xldat   = readmatrix([FilterPath, 'PRIMA_25um_Jochem_filters for tests'],'Sheet','S3168R15','Range','A2:B2384');%K2168 (or2186) HPE 1x 27.5
    m_4     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPFchip2= abs(xldat(:,2)); %6 THz HPF
    lowerdown = xldat(:,1) < 100; %affected by leakage in FTS
    HPFchip2(lowerdown) = lowT_outband; 
    clear xldat
    xldat   = readmatrix([FilterPath, 'PRIMA_25um_Jochem_filters for tests'],'Sheet','C0288_1','Range','A2:B12181');%FP3238 300cm-1 HPE, 3x 25.4 mm
    m_5     = 100*xldat(:,1); %convert to m-1, data in cm-1  %
    HPF     = abs(xldat(:,2)); %6 THz HPF
    clear xldat
    % 
    %Radiator: BP,LP,HP
    filterform{1}(:,1)=[0 m_1' ];           %BPF FP3223, adding low F point = lowest F datapoint
    filterform{1}(:,2)=[BPF(1) BPF' ];    
    filterform{2}(:,1)=[0 m_2' ];                   %LPF FP3253, adding low F point = lowest F datapoint
    filterform{2}(:,2)=[LPF(1) LPF']; %
    filterform{3}(:,1)=[0 m_5' ];           %HPF FP3238, adding low F point = 1e-4 according to Cardiff
    filterform{3}(:,2)=[lowT_outband HPF' ];    %%adding low F point = very low T as communicated with Cardiff.t 
    %LT box: BP,HP,HP
    filterform{4}(:,1)=[0 m_1' ];           %BPF FP3223, adding low F point = lowest F datapoint
    filterform{4}(:,2)=[BPF(1) BPF' ];    
    filterform{5}(:,1)=[0 m_5' ];           %HPF FP3238, adding low F point = 1e-4 according to Cardiff
    filterform{5}(:,2)=[lowT_outband HPF' ];    %%adding low F point = very low T as communicated with Cardiff.t 
    filterform{6}(:,1)=[0 m_5' ];           %HPF FP3238, adding low F point = 1e-4 according to Cardiff
    filterform{6}(:,2)=[lowT_outband HPF' ];    %%adding low F point = very low T as communicated with Cardiff.t 
    %Sample: BPF K2117, HPF K2168 and LPF K2478
    filterform{7}(:,1)=[0 m_1' ];           %BPF FP3223, adding low F point = lowest F datapoint
    filterform{7}(:,2)=[BPF(1) BPF' ];    
    filterform{8}(:,1)=[0 m_3' ];           %HPF FP3251, adding low F point = 1e-2 according to Cardiff
    filterform{8}(:,2)=[lowT_outband HPFchip' ];    %%adding low F point = very low T as communicated with Cardiff.t 


    legendstrr={...
        'BB: FP3223 25um BP','BB: FP3253 600cm-1 LPE','BB: %FP3238 300cm-1 HPE',...
        'LT Box: FP3223 25um BP','LT Box: FP3238 300cm-1 HPE','LT Box: FP3238 300cm-1 HPE',...
        'Sample: FP3223 25um BP','Sample: HPF FP3251'};
    if method.MaxFreq < 9e12
        error('Max Frequency in integartion < 9 Thz for 8 Thz filter stack');
    end

elseif isfile(method.filter)
    % We assume that a file is given,
    M = dlmread(method.filter,'\t');
    if size(M,2) == 2
        bba=flipdim(M,1);
        filterform{1}(:,1)=[bba(:,1)'*100 0];    % from cm-1 to m-1
        filterform{1}(:,2)=[bba(:,2)' 0];        % S21
        legendstrr=method.filter;
    else
        error('Filter file has wrong dimensions')
    end


else
    error('No good filter selection')
end

end
