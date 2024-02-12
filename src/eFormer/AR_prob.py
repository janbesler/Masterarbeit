# %%
class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class SeriesDecomp(nn.Module):
    """
    Series decomposition block for extracting trend and seasonal components.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

class SeasonalDecomp(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

""" 
class TriangularCausalMask():
    
    # Masking the future data points using a triangle.
    
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
"""


# %%
class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=False, factor=1, scale=None, attention_dropout=0.1, output_attention=False, top_k=2):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.top_k = top_k

    def time_delay_agg(self, values, corr):
        """
        Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = self.top_k
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)

        weights, delay = torch.topk(corr, self.top_k, dim=-1)
        # Reshape delay and weights dynamically
        delay = delay.reshape(batch, head, self.top_k, length)
        weights = weights.reshape(batch, head, self.top_k, length)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # print(f"tmp_corr non-zero:", len(tmp_corr[tmp_corr != 0]))
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(init_index).float()

        for i in range(self.top_k):
            tmp_delay = init_index.expand_as(delay) + delay[..., i].unsqueeze(-1)
            tmp_delay = torch.clamp(tmp_delay, min=0, max=length-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:,:,i,:].unsqueeze(1))
            
        # print(f"nonzero output: {len(delays_agg[delays_agg!=0])}")
        
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # Ensure corr has the shape [B, H, L, L]
        corr = corr.unsqueeze(2)
        expand_multiplier = L // corr.shape[3]
        corr = corr.expand(-1, -1, expand_multiplier, -1, -1)
        corr = corr.reshape(
            corr.shape[0],
            corr.shape[1],
            L,
            corr.shape[-1]
        )

        # Create and apply the mask
        if self.mask_flag:
            mask = TriangularCausalMask(B, L, device=queries.device).mask
            mask = mask.expand(-1, H, -1, -1)  # Shape: [B, H, L, L]
            corr = corr.masked_fill(mask, 0)  # Zero out the masked positions

        # time delay aggregation
        V = self.time_delay_agg(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        # time delay agg
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        #print(f"out shape: {self.out_projection(out).shape}")
        #print(f"attn shape: {attn.shape}")

        return self.out_projection(out), attn

# %%
class AutoCorrelationLayer(nn.Module):
    def __init__(self, auto_corr, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()
        # initialize AutoCorr
        self.auto_corr = auto_corr 

    def forward(self, queries, keys, values, attn_mask):
        # Use the provided AutoCorrelation instance
        return self.auto_corr(queries, keys, values, attn_mask)

class MyAutoCorrelationModel(nn.Module):
    def __init__(self, d_model, n_heads, B, L, input_dim, top_k=2, device="cpu"):
        super(MyAutoCorrelationModel, self).__init__()

        # Create the AutoCorrelationLayer with the AutoCorrelation instance
        self.n_heads = n_heads
        self.auto_corr = AutoCorrelation(top_k=top_k)
        self.auto_corr_layer = AutoCorrelationLayer(self.auto_corr, d_model, n_heads)
        self.top_k = top_k

    def forward(self, prob_embeddings):
        H = self.n_heads
        E = input_dim // H # size per head

        # Ensure that E is an even number
        if input_dim % H != 0:
            raise ValueError("Half the feature dimension is not divisible by the number of heads.")

        # Split tensor into means and variances
        means, variances = prob_embeddings.split(input_dim, dim=-1)

        # Reshape both halves for multi-head format: [B, L, H, E]
        reshaped_means = means.view(B, L, H, E)
        reshaped_variances = variances.view(B, L, H, E)

        # Concatenate the reshaped means and variances
        reshaped_embeddings = torch.cat([reshaped_means, reshaped_variances], dim=-1)

        return self.auto_corr_layer(reshaped_embeddings, reshaped_embeddings, reshaped_embeddings, None)[0]

# %%
class AutoCorrEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(AutoCorrEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class AutoCorrEncoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(AutoCorrEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


