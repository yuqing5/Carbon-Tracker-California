function f = rescaleImport(indices)
	m = loadcase('CaliforniaTestSystem.m');
	indices = cell2mat(indices);
	%remove pmax = zero
	%idx = find(m.gen(:,9)==0);
	%m.gen(idx,:) = [];
	%m.gencost(idx, :) = [];
	%remove solar
	%m.gencost(indices, 6) = m.gen(indices, 6)/2;
	for i = indices
		m.gencost(i, 6) = m.gencost(i, 6)*0.42;
	end
	f = savecase('CaliforniaTestSystem.m', m);
	%a = sum(m.bus(:,3));
	%f = fprintf('Power demand: %f/n', a);
end