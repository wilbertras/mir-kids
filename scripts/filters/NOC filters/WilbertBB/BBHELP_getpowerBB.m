% Tbb = A 1xN vector of temperatures of the blackbody.
% pad = path where the filterfiles are, can be left out, pad = pwd in that case. it will look in [pad,filesep,'filterfiles',filesep];%
% method. A struct containing information about the used setup including:
%         filters, lens size, and aperture determination. The following
%         struct elements are required:
%
%   method.pol = Number of polarizations used (equal to either 1 or 2)
%
%   method.filter = 12 THz              % 12 THz (24 um) 2022%
%   method.filter='GBLF'                % Groundbird low frequency (2020)
%   method.filter='8 THz'               % 7.8 THz Superkids stack (2020) - outdated, likely to be destroyed%
%   method.filter='7 THz'               % 7 THz Probes stack (2023)
%   method.filter='7 THz +1BP'          % 7 THz Probes stack (2023)  1 extra BP%
%   method.filter='7 THz -1BP'          % 7 THz Probes stack (2023)  1 less BP%

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
%OUTPUT:
% TotalPbb.             % 1xN vector of blackbody power received by the lens/antenna ateach blackbody temperature%
% method.Etendue.       % Optical throughput of the system at the center frequency of the band %
% method.centrefreq     % central frequency
% method.filterBW       % effective BW
% Filtertransmission.   %2xM vector of total filter transmission [frequency (Hz),transmission].
%                       frequency is given with the resolution specified in method.freqresolution  
% NEP.                  %struct containing four 1xN vectors that for each blackbody temperature give the expected NEP due to:%
%   NEP.g_r         %Generation Recombination Noise (using method.Delta)
%   NEP.poisson     %Poissonian photon noise
%   NEP.wave        %Wave Bunching Photon Noise
%   NEP.totphoton   %Total NEP due to Photon, GR and wavebunching noise.
%
%SUBROUTINES:
% getfilterform (included below)
% plotresults (included below)
%
%REQUIRED FILES (filters, located in /filterfiles subdir):
% W969 37cmLPESCUBAII.txt
% W1275_350GHz.dat
% totalLbandfilters.txt
% B386 18cm LPE.dat
% W1052 14cm LPE SCUBAII.dat
% H20coupling efficiency upto1_3THz.txt
% H10coupling efficiency.txt

%close all
%clear all
h = 6.6262e-34;		% J.s
e = 1.6e-19;        % C
% c = 2.9998e8;		% m/s
% k = 1.3806e-23;		% J/K
close all
clc
plotfiguresBBscript=1;
method.filter           = '12 THz';    %'GBLF', '8THz', '7 THz', '7 THz +1BP', '7 THz -1BP','1_6THz'   '850 GHz'  '650 GHz'  '350 GHz 4Filters' '1_6THz1file'=old one
method.tp               = 'lambda^2';       %'lambda^2' 'Geometrical'   
method.freqresolution   = 10; %in GHz
method.MaxFreq          = 25e12;
method.pol              = 1;               % one polarization
method.GR               = 4;              %R noise prefactor (4)
ndiv                    = 4;

Tbb = [3:0.5:300];% 40:20:300]';

%only Lambda^2
method.eta_c = 1;        %optical efficiency CST

%only geometrical
method.lensdiameter=1;%lens diameter in millimeters. 
method.opening_angle=10 ;

pad=pwd;

%
disp(method);
[TotalPbb,FilterTransmission,NEP,method]=blackbody_23(Tbb,method,plotfiguresBBscript,pad,1,ndiv);

MakeGoodFigureBB(18,14,11,[method.filter '_coupling' num2str(method.eta_c,'%0.2g') 'power_ori'])
%%
figure(1)
axes('LineWidth',2,'FontSize',16)
subplot(2,3,1)

loglog(FilterTransmission(1,:)/1e12,FilterTransmission(2,:),'k-');hold on
%
loglog([method.centrefreq-method.filterBW/2 method.centrefreq-method.filterBW/2 ...
    method.centrefreq+method.filterBW/2 method.centrefreq+method.filterBW/2]/1e12,...
    [1e-15 max(FilterTransmission(2,:)) max(FilterTransmission(2,:)) 1e-15],'-b')
loglog([method.centrefreq method.centrefreq]/1e12,...
    [1e-15  1.2*max(FilterTransmission(2,:))],'-r')
axis tight;
ylim([0 1.2*max(FilterTransmission(2,:))])
xlabel('Frequency (THz)')
ylabel('Filter Transmission')
grid on

title(['F0= ' num2str(round(method.centrefreq/1e9)) ' BW= ' num2str(round(method.filterBW/1e9)) ...
    ' GHz, Peak transm. = ' num2str(0.01*max(round(FilterTransmission(2,:)*100)))  ', QO coupling = ' num2str(0.01*max(round(method.eta_c*100)))])
hold off
%
subplot(2,3,2)

semilogx(FilterTransmission(1,:)/1e12,FilterTransmission(2,:),'k-','Linewidth',2);hold on
%
semilogx([method.centrefreq-method.filterBW/2 method.centrefreq-method.filterBW/2 ...
    method.centrefreq+method.filterBW/2 method.centrefreq+method.filterBW/2]/1e12,...
    [1e-15 max(FilterTransmission(2,:)) max(FilterTransmission(2,:)) 1e-15],'-b')
semilogx([method.centrefreq method.centrefreq]/1e12,...
    [1e-15  1*max(FilterTransmission(2,:))],'-r')
xlim([FilterTransmission(1,1) FilterTransmission(1,end)]/1e12);
ylim([0 1.2*max(FilterTransmission(2,:))])
xlabel('Frequency (THz)')
ylabel('Filter Transmission')
grid on

title(['F0= ' num2str(round(method.centrefreq/1e9)) ' BW= ' num2str(round(method.filterBW/1e9)) ...
    ' GHz, Peak transm. = ' num2str(0.01*max(round(FilterTransmission(2,:)*100)))  ', QO coupling = ' num2str(0.01*max(round(method.eta_c*100)))])
hold off

%
subplot(2,3,3)
semilogy(Tbb,TotalPbb*1e15,'k-','LineWidth',2)
xlabel('T (K)')
ylabel('P (fW)')
axis tight
hold off
grid on
%

subplot(2,3,4)
Photon_Energy = h*method.centrefreq;
semilogy(Tbb,TotalPbb/Photon_Energy,'k-','LineWidth',2)
xlabel('T (K)')
ylabel('Photon rate (1/s)')
title(['E_{photon} = ' num2str(1e3*Photon_Energy/e) ' meV'])
axis tight
hold off
grid on
%
NEP.photonrate = TotalPbb/Photon_Energy;

subplot(2,3,5)
LH = '';%num2str(round(method.centrefreq/1e9));
%Overview of the calculated NEP's use to photon induced pair breaking
semilogy(Tbb,NEP.poisson,['b'],'MarkerSize',8)
hold on
semilogy(Tbb,NEP.wave,['g'],'MarkerSize',8)
semilogy(Tbb,NEP.g_r,['r'],'MarkerSize',8)
semilogy(Tbb,NEP.totphoton, ['k'],'MarkerSize',8)
legend([LH 'Poisson'],[LH 'Bunching'],[LH 'Recombination'],[LH 'NEP_{BLIP}'],'Location','Best')
ylabel('NEP(P_{abs}) (W/\surd{Hz}Hz)')
xlabel('T  (K)')
title('NEP calc. overview @ detector')
axis tight;
ylim([1e-22 max(NEP.totphoton)])
grid on
%
subplot(2,3,6)
LH = '';%num2str(round(method.centrefreq/1e9));
%Overview of the calculated NEP's use to photon induced pair breaking
loglog(TotalPbb*1e15,NEP.poisson,['b'],'MarkerSize',8)
hold on
loglog(TotalPbb*1e15,NEP.wave,['g'],'MarkerSize',8)
loglog(TotalPbb*1e15,NEP.g_r,['r'],'MarkerSize',8)
loglog(TotalPbb*1e15,NEP.totphoton, ['k'],'MarkerSize',8)
legend([LH 'Poisson'],[LH 'Bunching'],[LH 'Recombination'],[LH 'NEP_{BLIP}'],'Location','Best')
ylabel('NEP(P_{abs}) [W/\surd{Hz}Hz]')
xlabel('P_{abs} (fW)')
title('NEP calc. overview @ detector')
grid on
axis tight;
%xlim([1e-3 max(TotalPbb*1e15)])
grid on



MakeGoodFigureBB(18,14,12,[method.filter '_coupling' num2str(method.eta_c,'%0.2g') 'power'])

%%
figure(3)

subplot(1,2,1)
plot(Tbb,TotalPbb*1e15,'k-','LineWidth',2)
xlabel('T (K)')
ylabel('P (fW)')
axis tight
hold off
grid on

%dPdT = ((TotalPbb(2:end) - TotalPbb(1:end-1))./(Tbb(2:end) - Tbb(1:end-1)))./TotalPbb(1:end-1);
dPdT = 50e-3*((TotalPbb(2:end) - TotalPbb(1:end-1))./TotalPbb(1:end-1))/(Tbb(2)-Tbb(1));
subplot(1,2,2)
plot(Tbb(2:end),dPdT,'k-','LineWidth',2)
xlabel('T (K)')
ylabel('dP/P for 50mK dT (1/50mK)')
axis tight
hold off
grid on

MakeGoodFigureBB(16,8,12,[method.filter '_dPdT' num2str(method.eta_c,'%0.2g') 'power'])

