% ==============================================================
% COPYRIGHT NOTICE
% ==============================================================
% 
% © [2024] [Budapest University of Technology and Economics]. All rights reserved.
%                [Department of Artificial Intelligence and Systems Engineering]
%      formally [Department of Measurement and Information Systems]
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% ==============================================================
% CITATION REQUEST
% ==============================================================
%
% If you use this code in your research, please cite the following paper:
%
% Al-Rikabi, H., Renczes, B.
% "Floating-Point Quantization Analysis of Multi-Layer Perceptron Artificial Neural Networks."
% Journal of Signal Processing Systems 96, 301–312 (2024).
% https://doi.org/10.1007/s11265-024-01911-0
% 
% ==============================================================

% Preliminaries
% 1- trainedANNexample is a file attached to this code of a neural network
% trained using matlab ANN training, the name of the model is net, it may
% contain any size of the activation function used in the paper. (newff)
% 2- The size of the floating point format can be changed below with p and expW

clear
clc
load trainedANNexample;
%% Floating-Point Quantizer
p=24 ; expW=8;     lenW=p+expW; 
q= quantizer('float','round',[lenW expW]);
%% Extracting Coefficients 
     L= length(net.layers);        %L = No. of Layers
     B= net.b;                           % Ectracting Biases
     LW= net.LW;                     % Ectracting Layers weights
     IW= net.IW;                       % Ectracting Input layer weights
N_in_o=quantize(q,x);              % Quantization of the input.
%% Normalization
in_range=net.inputs{:}.range;                                      % The range of the training
in_min= in_range(:,1);                                                  % dataset should be extracted
                                                                                     % here from the network variable,
in_max= in_range(:,2);                                                 % and quantizaed so that they will
                                                                                   % also be neglegted

f2=(in_max-in_min);                                                 % The input process has
K=2./f2;                                                                    % a K = [-1 to 1] / [input range]

f1=N_in_o-in_min;                                                % first operation in the normalization part;
                                                                          % subtracting in_min
f4=f1.*K;                                                            % Then multipling by K  
                                    
f5=f4-1;                                                 % then just subtracting one (adding y_min which is {-1})

N_in_Nor=f5;          % making that variable that always defines the input matrix to each neuron
                               % it will be updated after each layer

%----------- The same operations above will be repeated but with quantization-----------%
  N_in=  N_in_o ;

in_minq=quantize(q,in_min);
in_maxq=quantize(q,in_max);

qf2=(in_maxq-in_minq);    qf2=quantize(q,qf2);                         
Kq=2./f2;                     Kq=quantize(q,Kq);        

qf1=N_in-in_minq;     thf1=  0.18*(2^(-2*p))*((qf1.^2));      qf1=quantize(q,qf1);   
b_qf1=  -(in_minq-in_min).*ones(size(qf1));
% The th** variables all over the code refers to the theoretical QNP of the corresponding stage

qf4=qf1.*Kq;     thf4=(thf1.*Kq.^2)  + 0.18*(2^(-2*p)).*((qf4.^2));     qf44=quantize(q,qf4);  
                      b_qf4=(Kq-K).*qf1 + b_qf1.*Kq;
qf5=qf44-1;    thf5=thf4 + 0.18*(2^(-2*p))*((qf5.^2));  qf5=quantize(q,qf5); 
b_qf5= b_qf4;
N_in_Norq=qf5;     % qf5 is the output the quantizaed input process block which is similar to
                               % the f5 above but quantized 

z_th_qnp=0;
z_sim_qnp=0;

%% One neuron QNP Loop
for k=1:L                                                        % L = No. of Layers
     
     if k==1                                                      % Here the weights are combined to one
        W=IW{1,1};                                         %  variable that will be used inside the   
        qW=quantize(q,W);
     else                                                          %  loops...... also they are qunaitzed so that
        W=LW{k,k-1};                                   % they will be ignored in this calculations 
        qW=quantize(q,W);
     end                                 
     
     m=width(W);                                        % m defines the number of inputs to the current layer
     n= height(W);                                        % n is the number of neurons in this layer
     acfu=net.layers{k}.transferFcn;               % acfu is the AF of this layer

     N_out=zeros(n,length(x(1,:)));                 % Reseting the neuron output vectors and defining 
     N_outq=zeros(n,length(x(1,:)));                 % their new size.
         
          if k ~=1                                               % The thf5 is the QNP at the output of the previous
          thf5=thnuu;                                          % block (input process) it will be updated after the 
          b_qf5=b_nuu;                                        % first layer.
          end                                                        

thnu=zeros(n,length(x(1,:)));                        % reseting the QNP at the output of a neuron (size)
b_nu=zeros(n,length(x(1,:)));
     for j=1:n                                                   %  n = No. of Neurons in the Layer k
        
         Mul=zeros(m,length(x(1,:)));                 % reseting and size definition of the multiplication
         Mulq=zeros(m,length(x(1,:)));                % vector, which will be the input * weight
          
         for i=1:m                                                      % m is the no. of Multiplications in this neuron
                Mul(i,:)= N_in_Nor(i,:).*W(j,i);                % mult. input to the corresponding weight
                Mulq(i,:)= N_in_Norq(i,:).*qW(j,i);           Mulqq=quantize(q,Mulq);    
         end                                                                                               % here, thmul finds QNP
                thmul=(W(j,:).^2)*thf5 + sum( 0.18*(2^(-2*p))*((Mulq.^2)) );    % of all multiplications
          % thmul sums the QNP of the multiplications, that sum should be
          % done later after the SoE, but it is the same, so this is easier.
              d_w=qW(j,:)-W(j,:);     
          b_mul =    (d_w'.*ones(size(N_in_Nor))).*N_in_Nor + ...
              ((W(j,:)'.*ones(size(N_in_Nor)))).* b_qf5;


          SoE= sum(Mul,1);           
          SoEq= sum(Mulqq,1);  thsom=thmul + 0.18*(2^(-2*p))*((SoEq.^2));
          SoEqq=quantize(q,SoEq);
          b_soe= sum(b_mul);                                            %   b_soe= vecnorm(b_mul)

                b=B{k} ;                                                           % The bias vector is quantized here to
          Biased= SoE + b(j);                                                % neglect the coeffitient quantization
                bq=  quantize(q,b);  
          Biasedq= SoEqq + bq(j);    thbi=thsom + 0.18*(2^(-2*p))*((Biasedq.^2));
          Biasedqq=quantize(q,Biasedq);                 %Adding the Bias and quantizing the result.
         
          b_bis= b_soe + (bq(j)-b(j));

          %   Activation Function
              switch acfu

                  case 'purelin'                                % For each AF, there are two paths one is
                 N_out(j,:)=Biased;                          % for the normal BPANN and the other is
                 N_outq(j,:)=Biasedqq;                    % for the quantized one. so there will be
                 thnu(j,:)=thbi;                                   % 'N_out' and 'N_outq' inside each case.
                %The PureLin function is just G(z)=z (paper notation)
                %QNP(G)= QNP(z) , no added QNP.
                 b_nu(j,:)=b_bis;

                    case 'tansig'
                 Ag=(-2)*Biased;
                 Ae=exp(Ag);
                 As=Ae+1;
                 Ar=1./ As;
                 Agg=(2)*Ar;
                 Af=Agg-1;
                 N_out(j,:)= Af; 

                qAg=(-2)*Biasedqq;   thAg=thbi.*4;  Agq=quantize(q,qAg);   b_Ag= -b_bis.*2;     

                qAe=exp(Agq);  thAe=thAg.*((qAe.^2)) + 0.18*(2^(-2*p))*((qAe.^2));
                  Aeq=quantize(q,qAe); b_Ae=qAe.*b_Ag;
                qAs=Aeq+1;      thAs=thAe + 0.18*(2^(-2*p))*((qAs.^2)); 
                  Asq=quantize(q,qAs);   b_As=b_Ae;  
                qAr=1./ Asq;      thAr=((qAr.^4)).*thAs + 0.18*(2^(-2*p))*((qAr.^2)); 
                  Arq=quantize(q,qAr);    b_Ar=-b_As./(Asq.^2);
                Agg=(2)*Arq;     thAgg=thAr.*4;                                      
                 Agg= quantize(q,Agg);  b_Agg=b_Ar.*2;
                qAf=Agg-1;        thAf=thAgg + 0.18*(2^(-2*p))*((qAf.^2));         
                  qAf=quantize(q,qAf);      b_Af=  b_Agg;        
                 N_outq(j,:)= qAf;                                        
                thnu(j,:)=thAf;                                            
                b_nu(j,:)=b_Af;

                  case 'logsig'
                 Am=(-1)*Biased;                              
                 Ae=exp(Am);
                 As=Ae+1;
                 Ar=1./ As;
                 N_out(j,:)= Ar;
                                                                                                                                                      
                 Am=(-1)*Biasedqq;          b_Ag= -b_bis;          
                 Ae=exp(Am);   thAe=thbi.*(Ae.^2) + 0.18*(2^(-2*p))*(Ae.^2);  
                    Ae=quantize(q,Ae);      b_Ae=Ae.*b_Ag;
                 As=Ae+1;        thAs=thAe + 0.18*(2^(-2*p))*((As.^2));     
                    As=quantize(q,As);      b_As=b_Ae;
                 Ar=1./ As;        thAr= (Ar.^4).*thAs + 0.18*(2^(-2*p))*((Ar.^2)); 
                    Ar=quantize(q,Ar);        b_Ar=-b_As./(As.^2);
                 N_outq(j,:)= Ar; 
                 thnu(j,:)=thAr;
                 b_nu(j,:)=b_Ar;

                    case 'radbas'
                  Ap=Biased.*Biased;
                  Am=(-1)*Ap;
                  Ae=exp(Am);
                  N_out(j,:)= Ae;         

                 Ap=Biasedqq.*Biasedqq;  thAp=(4.*Ap.*thbi) + 0.18*(2^(-2*p))*((Ap.^2));
                       Ap=quantize(q,Ap);       b_Ap=2*Ap.*b_bis./Biasedqq;
                  Am=(-1)*Ap;                        b_Am=b_Ap.*(-1);
                  Ae=exp(Am);               thAe=thAp.*((Ae.^2)) + 0.18*(2^(-2*p))*((Ae.^2)); 
                      Ae=quantize(q,Ae);         b_Ae=Ae.*b_Am;
                  N_outq(j,:)= Ae;
                  thnu(j,:)=thAe;
                 b_nu(j,:)=b_Ae;

                    case 'tribas'

                      A_abs=abs(Biased);
                      A_1minus=1-A_abs;
                      A_max=max(A_1minus,0);
                      N_out(j,:)= A_max;               
                      A_absq=abs(Biasedqq);                                  
                  ii=size(Biasedqq);
                 ii=ii(2);
                 b_abs=zeros(size(Biasedqq));
                for z= 1:ii
                    if Biasedqq(1,z) < 0
                       b_abs(1,z)=-b_bis(1,z);
                    elseif Biasedqq(1,z) > 0
                      b_abs(1,z)=b_bis(1,z);
                    else
                      b_abs(1,z)=0;
                    end
                end 

                      A_1minusq=1-A_absq;                  b_1minus=-b_abs;                                                                                         
    th_1minus= thbi + 0.18*(2^(-2*p))*((A_1minusq.^2)); A_1minusq=quantize(q,A_1minusq);     
                      A_maxq=max(A_1minusq,0);    
                      N_outq(j,:)= A_maxq;   

                                                                                                                
                 b_A_max=zeros(size(A_1minusq));
                 thnuA_max=zeros(size(A_1minusq));
                 ii=size(A_1minusq);
                 ii=ii(2);
                for z= 1:ii
                    if A_1minusq(1,z) <= 0
                       b_A_max(1,z)=0;
                       thnuA_max(1,z)=0;
                    else
                      b_A_max(1,z)=b_1minus(1,z);
                      thnuA_max(1,z)=th_1minus(1,z);
                    end
                end
                 b_nu(j,:)=b_A_max;
                thnu(j,:)=thnuA_max;
                           
                    case 'hardlim'
                  Ah=hardlim(Biased);          
                  N_out(j,:)= Ah;

                  Ah=hardlim(Biasedqq);     
                  N_outq(j,:)= Ah;

                 thnu(j,:)=0;                   
                  b_nu(j,:)=0;

                  case 'hardlims'
                  Ahs=hardlims(Biased);
                  N_out(j,:)= Ahs;
                  Ahsq=hardlims(Biasedqq);    
                  N_outq(j,:)= Ahsq;
                  thnu(j,:)=0;   
                  b_nu(j,:)=0;

                    case 'poslin'
                  Ap=poslin(Biased);          % poslin does not generate new QNP, but it can pass
                  N_out(j,:)= Ap;                 % theough input QNP when the input is positive.

                  Ap=poslin(Biasedqq);  
                  N_outq(j,:)= Ap;

                 ii=size(Biasedqq);            % Elementwise IF statement
                 ii=ii(2);
                 for z= 1:ii
                    if Biasedqq(1,z) <= 0
                       b_nu(j,z)=0;
                       thnu(j,z)=0;   
                    else
                       b_nu(j,z)= b_bis(1,z);
                       thnu(j,z)=thbi(1,z);   
                    end
                 end

                    case 'satlin'
                  Ast=satlin(Biased);
                  N_out(j,:)= Ast;

                  Ast=satlin(Biasedqq);         
                  N_outq(j,:)= Ast;           % no rounding  just If below
 

                 ii=size(Biasedqq);
                 ii=ii(2);
                 b_Ast=zeros(size(Biasedqq));
                 th_Ast=zeros(size(Biasedqq));
                for z= 1:ii
                    if (Biasedqq(1,z)<=0) || (Biasedqq(1,z)>=1)
                        b_Ast(1,z)=0;
                        th_Ast(1,z)=0;
                    else
                        b_Ast(1,z)=b_bis(1,z);
                        th_Ast(1,z)=thbi(1,z);
                    end
                end
                  thnu(j,:)=th_Ast;           
                  b_nu(j,:)=b_Ast;

                  case 'satlins'
                  Asts=satlins(Biased);
                  N_out(j,:)= Asts;

                  Asts=satlins(Biasedqq);             
                  N_outq(j,:)= Asts;                  %also no rounding just If below

                  ii=size(Biasedqq);
                 ii=ii(2);
                 b_Ast=zeros(size(Biasedqq));
                 th_Ast=zeros(size(Biasedqq));
                for z= 1:ii
                    if (Biasedqq(1,z)<=-1) || (Biasedqq(1,z)>=1)
                        b_Ast(1,z)=0;
                        th_Ast(1,z)=0;
                    else
                        b_Ast(1,z)=b_bis(1,z);
                        th_Ast(1,z)=thbi(1,z);
                    end
                end

                  thnu(j,:)=th_Ast;           
                  b_nu(j,:)=b_Ast;

              %Here the swithc ends
              end

     z_th_qnp(j,k)=mean(thnu(j,:));
     z_sim_qnp(j,k)=mean((N_outq(j,:)-N_out(j,:)).^2);
     %Here the neuron loop ends            
     end    
szN_out=size(N_out);                 % Connecting the output of the neuron to the input of
N_in_Nor=N_out;                         % the neurons of the next layer. there are resets at the
N_in_Norq=N_outq;                      % bigening of the loops
thnuu=thnu;                                    % this line is added to save QNP to go to output process 
thnu=zeros;                                      % reset QNP output of the neurons.
b_nuu=b_nu;
end

%% DeNormalization
out_range=net.outputs{L}.range;
out_min= out_range(:,1);     
out_max= out_range(:,2);     

ff2= (out_max-out_min);     
ff3=ff2./2;                          % ff3 represents K similar to the one in the input process 
ff1=N_out+1; 
ff4=ff1.*ff3;
ff5= ff4 +out_min;
out=ff5;                                  % out represents the normal output of the entire ANN

out_minq=quantize(q,out_min);out_maxq=quantize(q,out_max);

ff2q= (out_maxq-out_minq);     ff2q=quantize(q,ff2q);
ff3q=ff2./2;                            ff3q=quantize(q,ff3q);     


qff1=N_outq+1;      thff1= thnuu + 0.18*(2^(-2*p))*((qff1.^2));     ff11=quantize(q,qff1); 
b_ff1=   b_nuu;
qff4=ff11.*ff3;        thff4= (thff1.*ff3.^2) + 0.18*(2^(-2*p))*((qff4.^2));    ff44=quantize(q,qff4); 
b_ff4=  (ff3q-ff3).*ff11 + b_ff1.*ff3;
qff5= ff44 +out_minq;    thff5= thff4 + 0.18*(2^(-2*p))*((qff5.^2));              ff55=quantize(q,qff5); 
b_ff5=b_ff4 + (out_minq-out_min) ;
outq=ff55;                                  % outq represents the quantized output of the entire ANN
%% Results and  Analysis
Bias_error=b_ff5;
Bias_error2=Bias_error.^2;
th_QNP=thff5;
segma=sqrt(th_QNP);                   % calculating Standard Deviation at each sample from
                                                        % the  theoretical  QNP

round_error_ub=(outq-Bias_error)-out; % this is the actual roundoff error not the approximated 
confidence_cuve=round_error_ub./segma;% here dividing the roundoff error by the s. deviation

 Mean_th_QNP=mean(th_QNP')                                        % mean of the th. QNP vector
 Mean_Square_Roundoff_Error=mean(round_error_ub.^2)  % mean square of actual r. error
round_error_bad=(outq)-out; 
confidence_cuve2=round_error_bad./segma;   

figure
h1 = confidence_cuve;
h2 = confidence_cuve2;
[counts1, binCenters1] = hist(h1, 50);
[counts2, binCenters2] = hist(h2, 50);
plot(binCenters1, counts1, 'r-o');
hold on;
plot(binCenters2, counts2, 'k-*');
ylabel('Number of occurrences')
xlabel( '$$\nu  / \sqrt{QNP}$$','Interpreter','latex')
grid on;grid minor;
legend1 = sprintf('CQE Neutralized' );
legend2 = sprintf('CQE Effect');
legend({legend1, legend2});


