function f = myFuncNoModi(total_dem)
	total_dem = double(total_dem);
	m = loadcase('CaliforniaTestSystem.m');
	m.bus(:,3) = (total_dem/sum(m.bus(:,3)))*m.bus(:,3);
	%U = unique(m.gen(:,2));
	m.gen(:,2) = (total_dem/sum(m.gen(:,2)))*m.gen(:,2);
	m.bus(:,4) = (total_dem/sum(m.bus(:,4)))*m.bus(:,4);
	f = savecase('CaliforniaTestSystem2.m', m);
	%a = sum(m.bus(:,3));
	%f = fprintf('Power demand: %f/n', a);
end