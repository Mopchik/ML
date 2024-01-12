import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, dim_embedding=32, dim_hidden=100, device='cpu'):
        super().__init__()
        self.embedding = torch.nn.Embedding(30000, dim_embedding)
        self.Wh = nn.Parameter(0.01 * torch.randn((dim_hidden + dim_embedding, dim_hidden)))
        self.bh = nn.Parameter(torch.randn((dim_hidden)))
        self.Wo = nn.Parameter(0.1 * torch.randn((dim_hidden, 1)))
        self.bo = nn.Parameter(torch.randn((1)))
        self.dim_hidden = dim_hidden
        self.device = device

    def forward(self, data):
        h = torch.zeros((len(data), self.dim_hidden)).to(self.device)
        data = self.embedding(data)

        words_count = len(data[0])
        for i in range(words_count):
            x = data[:, i, :]
            change_mask = ((x != data[:, words_count - 1, :]).any(dim=1))
            h[change_mask] = torch.sigmoid(torch.cat((x[change_mask], h[change_mask]), 1) @ self.Wh + self.bh)
        output = torch.sigmoid(h @ self.Wo + self.bo)
        return output


class ConvNN(nn.Module):
    def __init__(self, words_count, dim_hidden):
        super().__init__()
        self.embedding = torch.nn.Embedding(30000, dim_hidden)
        self.first = nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden // 4, kernel_size=5, padding=2, stride=1)
        self.second = nn.Conv1d(in_channels=dim_hidden // 4, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.third = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=2)
        self.fc = nn.Linear(in_features=words_count // 4 + 1, out_features=1)
        self.words_count = words_count
        self.dim_hidden = dim_hidden

    def forward(self, data):
        data = self.embedding(data)
        data = data.permute(0, 2, 1)
        data = self.first(data)
        data = self.second(data)
        data = self.third(data)
        data = data.view(data.size(0), -1)
        output = torch.sigmoid(self.fc(data))
        return output


class LSTM(nn.Module):
    def __init__(self, dim_embedding=32, dim_hidden=100, device='cpu'):
        super().__init__()
        self.embedding = torch.nn.Embedding(30000, dim_embedding)
        self.Wfh = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bfh = nn.Parameter(torch.randn((dim_hidden)))
        self.Wih = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bih = nn.Parameter(torch.randn((dim_hidden)))
        self.Woh = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.boh = nn.Parameter(torch.randn((dim_hidden)))
        self.Wch = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bch = nn.Parameter(torch.randn((dim_hidden)))
        self.bo = nn.Parameter(torch.randn((1)))
        self.Wo = nn.Parameter(0.1 * torch.randn((dim_hidden, 1)))
        self.dim_hidden = dim_hidden
        self.device = device

    def forward(self, data):
        h = torch.zeros((len(data), self.dim_hidden)).to(self.device)
        context = torch.zeros((len(data), self.dim_hidden)).to(self.device)
        data = self.embedding(data)

        words_count = len(data[0])

        for i in range(words_count):
            x = data[:, i, :]
            change_mask = (x != data[:, words_count - 1, :]).any(dim=1)
            xh = torch.cat((x[change_mask], h[change_mask]), 1)
            f = torch.sigmoid((xh @ self.Wfh + self.bfh))
            i = torch.sigmoid((xh @ self.Wih + self.bih))
            o = torch.sigmoid((xh @ self.Woh + self.boh))
            c = torch.tanh((xh @ self.Wch + self.bch))
            context[change_mask] = context[change_mask] * f + i * c
            h[change_mask] = o * torch.tanh(context[change_mask])
        output = torch.sigmoid(h @ self.Wo + self.bo)
        return output


class LSTMnLayers(nn.Module):
    def __init__(self, dim_embedding=32, dim_hidden=100, n_layers=2, device='cpu'):
        super().__init__()
        self.embedding = torch.nn.Embedding(30000, dim_embedding)

        self.parametersList = nn.ParameterList()
        self.parameter_dict = {}
        for i in range(n_layers):
            if i == 0:
                dimParam = dim_embedding + dim_hidden
            else:
                dimParam = 2 * dim_hidden
            Wfh = nn.Parameter(0.1 * torch.randn((dimParam, dim_hidden)))
            self.parameter_dict[f'Wfh{i}'] = Wfh
            self.parametersList.append(Wfh)
            bfh = nn.Parameter(torch.randn((dim_hidden)))
            self.parameter_dict[f'bfh{i}'] = bfh
            self.parametersList.append(bfh)
            Wih = nn.Parameter(0.1 * torch.randn((dimParam, dim_hidden)))
            self.parameter_dict[f'Wih{i}'] = Wih
            self.parametersList.append(Wih)
            bih = nn.Parameter(torch.randn((dim_hidden)))
            self.parameter_dict[f'bih{i}'] = bih
            self.parametersList.append(bih)
            Woh = nn.Parameter(0.1 * torch.randn((dimParam, dim_hidden)))
            self.parameter_dict[f'Woh{i}'] = Woh
            self.parametersList.append(Woh)
            boh = nn.Parameter(torch.randn((dim_hidden)))
            self.parameter_dict[f'boh{i}'] = boh
            self.parametersList.append(boh)
            Wch = nn.Parameter(0.1 * torch.randn((dimParam, dim_hidden)))
            self.parameter_dict[f'Wch{i}'] = Wch
            self.parametersList.append(Wch)
            bch = nn.Parameter(torch.randn((dim_hidden)))
            self.parameter_dict[f'bch{i}'] = bch
            self.parametersList.append(bch)

        self.bo = nn.Parameter(torch.randn((1)))
        self.Wo = nn.Parameter(0.1 * torch.randn((dim_hidden, 1)))
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.device = device

    def forward(self, data):
        h = []
        context = []
        for i in range(self.n_layers):
            h.append(torch.zeros((len(data), self.dim_hidden)).to(self.device))
            context.append(torch.zeros((len(data), self.dim_hidden)).to(self.device))

        data = self.embedding(data)

        words_count = len(data[0])
        for i in range(words_count):
            x = data[:, i, :]
            change_mask = (x != data[:, words_count - 1, :]).any(dim=1)
            for j in range(self.n_layers):
                xh = torch.cat((x[change_mask], h[j][change_mask]), 1)
                f = torch.sigmoid(xh @ self.parameter_dict[f'Wfh{j}'] + self.parameter_dict[f'bfh{j}'])
                i = torch.sigmoid(xh @ self.parameter_dict[f'Wih{j}'] + self.parameter_dict[f'bih{j}'])
                o = torch.sigmoid(xh @ self.parameter_dict[f'Woh{j}'] + self.parameter_dict[f'boh{j}'])
                c = torch.tanh(xh @ self.parameter_dict[f'Wch{j}']+ self.parameter_dict[f'bch{j}'])
                context[j][change_mask] = context[j][change_mask] * f + i * c
                h[j][change_mask] = o * torch.tanh(context[j][change_mask])
                x = h[j]

        output = torch.sigmoid(h[-1] @ self.Wo + self.bo)
        return output


class LSTMbiDir(nn.Module):
    def __init__(self, dim_embedding=32, dim_hidden=100, device='cpu'):
        super().__init__()
        self.embedding = torch.nn.Embedding(30000, dim_embedding)

        self.Wfh1 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bfh1 = nn.Parameter(torch.randn((dim_hidden)))
        self.Wih1 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bih1 = nn.Parameter(torch.randn((dim_hidden)))
        self.Woh1 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.boh1 = nn.Parameter(torch.randn((dim_hidden)))
        self.Wch1 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bch1 = nn.Parameter(torch.randn((dim_hidden)))

        self.Wfh2 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bfh2 = nn.Parameter(torch.randn((dim_hidden)))
        self.Wih2 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bih2 = nn.Parameter(torch.randn((dim_hidden)))
        self.Woh2 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.boh2 = nn.Parameter(torch.randn((dim_hidden)))
        self.Wch2 = nn.Parameter(0.1 * torch.randn((dim_embedding + dim_hidden, dim_hidden)))
        self.bch2 = nn.Parameter(torch.randn((dim_hidden)))

        self.bo = nn.Parameter(torch.randn((1)))
        self.Wo = nn.Parameter(0.1 * torch.randn((2 * dim_hidden, 1)))
        self.dim_hidden = dim_hidden
        self.device = device

    def forward(self, data):
        h1 = torch.zeros((len(data), self.dim_hidden)).to(self.device)
        h2 = torch.zeros((len(data), self.dim_hidden)).to(self.device)
        context1 = torch.zeros((len(data), self.dim_hidden)).to(self.device)
        context2 = torch.zeros((len(data), self.dim_hidden)).to(self.device)

        data = self.embedding(data)

        words_count = len(data[0])

        for i in range(words_count):
            x = data[:, i, :]
            change_mask = (x != data[:, words_count - 1, :]).any(dim=1)
            xh = torch.cat((x[change_mask], h1[change_mask]), 1)
            f = torch.sigmoid((xh @ self.Wfh1 + self.bfh1))
            i = torch.sigmoid((xh @ self.Wih1 + self.bih1))
            o = torch.sigmoid((xh @ self.Woh1 + self.boh1))
            c = torch.tanh((xh @ self.Wch1 + self.bch1))
            context1[change_mask] = context1[change_mask] * f + i * c
            h1[change_mask] = o * torch.tanh(context1[change_mask])

        for i in range(words_count - 1, -1, -1):
            x = data[:, i, :]
            change_mask = (x != data[:, words_count - 1, :]).any(dim=1)
            xh = torch.cat((x[change_mask], h2[change_mask]), 1)
            f = torch.sigmoid((xh @ self.Wfh2 + self.bfh2))
            i = torch.sigmoid((xh @ self.Wih2 + self.bih2))
            o = torch.sigmoid((xh @ self.Woh2 + self.boh2))
            c = torch.tanh((xh @ self.Wch2 + self.bch2))
            context2[change_mask] = context2[change_mask] * f + i * c
            h2[change_mask] = o * torch.tanh(context2[change_mask])

        output = torch.sigmoid(torch.cat((h1, h2), 1) @ self.Wo + self.bo)
        return output