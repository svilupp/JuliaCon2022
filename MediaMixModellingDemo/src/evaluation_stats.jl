# Pseudo Rsquared: Calculates ratio of variance against the predictions vs against the mean value 
function pseudor2(y_true::Vector{T}, y_pred::Vector{T}) where {T <: Real}
    1 - sum((y_true .- y_pred) .^ 2) / sum((y_true .- mean(y_true)) .^ 2)
end
function rmse(y_true::Vector{T}, y_pred::Vector{T}) where {T <: Real}
    sqrt(mean((y_true .- y_pred) .^ 2))
end
function nrmse(y_true::Vector{T}, y_pred::Vector{T}) where {T <: Real}
    rmse(y_true, y_pred) / (maximum(y_true) - minimum(y_true))
end
