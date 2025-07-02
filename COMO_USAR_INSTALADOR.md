# Como Usar o Instalador Windows do CardioAI Pro

## 📋 Pré-requisitos
- Windows 10 ou 11 (64-bit)
- Conexão com internet (para download automático do Node.js se necessário)

## 🚀 Instruções de Uso

### Passo 1: Baixar o Projeto
1. Baixe ou clone o projeto CardioAI Pro
2. Extraia os arquivos se necessário

### Passo 2: Executar o Instalador
**NOVO**: Agora você pode executar o instalador de qualquer forma:

#### Opção 1: Duplo Clique (Mais Fácil) ✅
1. Navegue até a pasta `cardio.ai.pro\windows_installer`
2. **Duplo clique** em `build_backend.py` e depois em `build_frontend.py`
3. Por fim, execute `cardioai_installer.nsi` com o NSIS para gerar o instalador

#### Opção 2: Prompt de Comando
1. Abra o **Prompt de Comando** (cmd) ou **PowerShell**
2. Navegue até a pasta do instalador:
   ```cmd
   cd caminho\para\cardio.ai.pro\windows_installer
   ```
3. Execute os scripts de build e compile o instalador:
   ```cmd
   python build_backend.py
   python build_frontend.py
   makensis cardioai_installer.nsi
   ```

**✅ IMPORTANTE**: O instalador agora funciona independentemente de onde você o executa!

## ✅ O que o Instalador Faz Automaticamente

### 1. **Detecção Automática de Diretório** 🆕
- ✅ **Detecta automaticamente** onde o script está localizado
- ✅ **Navega automaticamente** para o diretório correto
- ✅ **Funciona quando executado por duplo clique** (não mais erro de System32)
- ✅ **Funciona de qualquer local** - linha de comando ou duplo clique

### 2. **Verificação de Dependências**
- ✅ Verifica se Python 3.8+ está instalado
- ✅ Verifica se Node.js está disponível
- ✅ **Baixa Node.js automaticamente** se não encontrado
- ✅ Verifica se NSIS está disponível (opcional)

### 3. **Build do Backend**
- ✅ Instala Poetry automaticamente se necessário
- ✅ Instala dependências Python
- ✅ Cria executável standalone (`cardioai-backend.exe`)
- ✅ Gera script de inicialização (`start_backend.bat`)

### 4. **Build do Frontend**
- ✅ Usa Node.js portable ou do sistema
- ✅ Instala dependências npm automaticamente
- ✅ Compila aplicação React/Vite
- ✅ Cria servidor frontend (`serve_frontend.py`)

### 5. **Criação do Instalador**
- ✅ Prepara todos os arquivos necessários
- ✅ Cria ícone válido automaticamente
- ✅ Gera arquivo de licença
- ✅ Compila instalador NSIS (se disponível)

## 🎯 Resultado Final

Após a execução bem-sucedida, você terá:

### Se NSIS estiver disponível:
- `CardioAI-Pro-Installer.exe` - **Instalador final para distribuição**

### Se NSIS não estiver disponível:
- `cardioai-backend.exe` - Executável do backend

### Instalador Unificado (Recomendado):
- `CardioAI-Pro-1.0.0-installer.exe` - **Instalador NSIS com backend e frontend** (229 MB)
- Localização: `frontend/dist-electron/CardioAI-Pro-1.0.0-installer.exe`
- Uso: Execute como administrador e siga o assistente de instalação
- `frontend_build/` - Arquivos do frontend
- `serve_frontend.py` - Servidor do frontend
- `cardioai_installer.nsi` - Script NSIS (para compilação manual)

## 🔧 Solução de Problemas

### ✅ RESOLVIDO: Erro de Diretório
**Problema Anterior**: "Please run this script from the project root or windows_installer directory"
**Status**: ✅ **CORRIGIDO** - O instalador agora detecta automaticamente o diretório correto

### Erro: "Python is not installed"
**Solução**: 
1. Baixe Python 3.8+ de https://python.org
2. Durante a instalação, marque "Add Python to PATH"
3. Reinicie o Prompt de Comando

### Erro: "Node.js is not installed"
**Solução**: O instalador baixará automaticamente! Se falhar:
1. Baixe Node.js LTS de https://nodejs.org
2. Durante a instalação, marque "Add to PATH"
3. Reinicie o computador

### Erro: "NSIS is not installed"
**Solução**: NSIS é opcional. Para instalá-lo:
1. Baixe NSIS de https://nsis.sourceforge.io/
2. Instale e adicione ao PATH
3. Execute o script novamente

## 🆕 Melhorias na Versão Atual

### ✅ Navegação Robusta de Diretório
- **Antes**: Falhava quando executado por duplo clique (System32)
- **Agora**: Usa `%~dp0` para detectar localização do script automaticamente
- **Resultado**: Funciona independentemente de como é executado

### ✅ Mensagens de Erro Melhoradas
- **Antes**: Mensagens confusas sobre diretório
- **Agora**: Instruções claras e soluções específicas
- **Resultado**: Usuários sabem exatamente o que fazer

### ✅ Experiência do Usuário Simplificada
- **Antes**: Usuário precisava navegar manualmente para diretório correto
- **Agora**: Duplo clique funciona diretamente
- **Resultado**: Instalação mais intuitiva e amigável

## 📞 Suporte

Se encontrar problemas:
1. ✅ **Duplo clique funciona agora** - tente primeiro esta opção
2. Certifique-se de ter permissões de administrador
3. Verifique sua conexão com internet
4. Consulte os arquivos de log gerados durante o build

## 🎉 Distribuição

### Opções de Distribuição:

**Opção 1: Instalador Unificado (Recomendado)**
O arquivo `CardioAI-Pro-1.0.0-installer.exe` pode ser distribuído diretamente. Usuários só precisam:
1. Baixar o arquivo instalador
2. Executar como administrador
3. Seguir o assistente de instalação
4. Lançar pelo atalho da área de trabalho

**Opção 2: Instalador NSIS**
O arquivo `CardioAI-Pro-Installer.exe` pode ser distribuído para usuários finais. Eles só precisam:
1. Executar o instalador
2. Seguir as instruções na tela
3. Usar o CardioAI Pro instalado

**Não há necessidade de instalação manual de dependências para usuários finais!**

---

## 📝 Notas Técnicas

### Detecção de Diretório
O instalador usa a variável `%~dp0` do Windows que contém o caminho completo do diretório onde o arquivo .bat está localizado, independentemente do diretório de trabalho atual. Isso resolve o problema onde duplo clique executava o script a partir de `C:\Windows\System32`.

### Compatibilidade
- ✅ Windows 10/11 (64-bit)
- ✅ Execução por duplo clique
- ✅ Execução via linha de comando
- ✅ Funciona de qualquer diretório de trabalho
