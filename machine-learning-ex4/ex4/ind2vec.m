function [vector]=ind2vec(ind)
  # Converts indices to vectors
  #
  #
  vectors = length(ind);
  vector = sparse(ind,1:vectors,ones(1,vectors));
endfunction
Back to Top
