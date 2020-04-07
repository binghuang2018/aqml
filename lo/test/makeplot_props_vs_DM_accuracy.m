

% now props vs. rcutoff in DM

figure(1); 
props = {'N', 'dipole (Debye)', 'Totel E (kcal/mol)', 'HOMO (eV)', 'LUMO (eV)'};

for i=1:5
  subplot(2,5,i); plot(rs, dp1(:,i), 'LineWidth',2); 
  xlabel('rc [A]'); ylabel(props{i});
  %set(gca,'color','none', 'FontName', 'Arial'); 
  ax = gca;
  ax.LineWidth = 1.5;
  ax.TickLength = [0.03 0.03];
  %ax.FontSize = 16;
  ax.Color = 'none';
  ax.FontName = 'Arial';
end

for i=1:5
  subplot(2,5,i+5); plot(rs, dp2(:,i), 'LineWidth',2); 
  xlabel('rc [A]'); ylabel(props{i});
  %set(gca,'color','none', 'FontName', 'Arial'); 
  ax = gca;
  ax.LineWidth = 1.5;
  ax.TickLength = [0.03 0.03];
  %ax.FontSize = 16;
  ax.Color = 'none';
  ax.FontName = 'Arial';
end





% now props vs. decimal

figure(2)

decimals = [2,3,4,5,6]; 
dp1 = [[-0.1152,  4.2968, -179.0051, -0.5898, -0.5960],
       [ 0.0026,  0.0981, -12.0520,  0.0137,  0.0136],
       [ 0.0002,  0.0083, -1.5248,  0.0012,  0.0012],
       [-0.0000,  0.0015,  0.0532, -0.0003, -0.0003],
       [ 0.0000,  0.0003, -0.0162,  0.0000, -0.0000]];

dp2 = [[ 0.0266,  0.8828,  270.7615,  0.2413,  0.1992],
       [ 0.0127,  0.4221, -11.7086,  0.0577,  0.0582],
       [-0.0003,  0.0113,  2.1883, -0.0018, -0.0020],
       [ 0.0000,  0.0013, -0.1501,  0.0003,  0.0002],
       [ 0.0000,  0.0001, -0.0038,  0.0000,  0.0000]];

for i=1:5
  subplot(2,5,i); plot(decimals, dp1(:,i), 'LineWidth',2); 
  xlabel('decimals'); ylabel(props{i});
  %set(gca,'color','none', 'FontName', 'Arial');
  ax = gca;
  ax.LineWidth = 1.5;
  ax.TickLength = [0.03 0.03];
  %ax.FontSize = 16;
  ax.Color = 'none';
  ax.FontName = 'Arial';
end

for i=1:5
  subplot(2,5,i+5); plot(decimals, dp2(:,i), 'LineWidth',2);
  xlabel('decimals'); ylabel(props{i});
  %set(gca,'color','none', 'FontName', 'Arial');
  ax = gca;
  ax.LineWidth = 1.5;
  ax.TickLength = [0.03 0.03];
  %ax.FontSize = 16;
  ax.Color = 'none';
  ax.FontName = 'Arial';
end


