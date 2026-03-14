"""
    log_growth_matrix(dataset::Dict{String, DataFrame}, firms::Array{String,1};
        Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0,
        testfirm = "AAPL", keycol::Symbol = :volume_weighted_average_price) -> Array{Float64,2}

Compute the excess log growth matrix for a set of firms:

    μ_{t,t-1}(r_f) = (1/Δt) * ln(S_t / S_{t-1}) - r_f

where S_t is the volume-weighted average price at time t, Δt is the time step
in years (default: 1/252 = one trading day), and r_f is the annual risk-free rate.

Rows = time periods, columns = firms (ordered by `firms` array).
"""
function log_growth_matrix(dataset::Dict{String, DataFrame}, 
    firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
    testfirm="AAPL", keycol::Symbol = :volume_weighted_average_price)::Array{Float64,2}

    number_of_firms = length(firms);
    number_of_trading_days = nrow(dataset[testfirm]);
    return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

    for i ∈ eachindex(firms) 
        firm_index = firms[i];
        firm_data = dataset[firm_index];

        for j ∈ 2:number_of_trading_days
            S₁ = firm_data[j-1, keycol];
            S₂ = firm_data[j, keycol];
            return_matrix[j-1, i] = (1/Δt)*(log(S₂/S₁)) - risk_free_rate;
        end
    end

    return return_matrix;
end
