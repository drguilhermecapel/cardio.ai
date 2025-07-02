# CardioAI Pro - Instalação Standalone

## Visão Geral

Esta versão standalone do CardioAI Pro foi projetada para usuários que desejam uma instalação simples e direta, sem necessidade de conhecimentos técnicos ou configurações complexas.

## Requisitos do Sistema

- **Sistema Operacional:** Windows 7 ou superior (64-bit)
- **Memória RAM:** Mínimo 4GB, recomendado 8GB
- **Espaço em Disco:** 500MB livres
- **Navegador:** Chrome, Firefox, Edge ou Safari (versões recentes)
- **Privilégios:** Administrador (apenas durante a instalação)

## Instalação Simples

### Opção 1: Instalação Automática (Recomendada)

1. **Baixe o instalador:**
   - `CardioAI-Pro-v1.0.0-Setup.exe`

2. **Execute a instalação:**
   - Clique duas vezes no arquivo
   - Clique em "Sim" quando o Windows solicitar privilégios de administrador
   - Aguarde a conclusão da instalação

### Opção 2: Instalação Manual

1. **Baixe o instalador:**
   - `CardioAI-Pro-v1.0.0-Setup.exe`

2. **Execute o instalador:**
   - Clique com o botão direito no arquivo
   - Selecione "Executar como administrador"
   - Siga as instruções na tela

### Opção 3: Versão Portátil

1. **Baixe o arquivo:**
   - `CardioAI-Pro-v1.0.0-Portable.zip`

2. **Extraia e execute:**
   - Extraia todos os arquivos para uma pasta
   - Clique duas vezes em `CardioAI-Pro.bat`
   - O sistema iniciará automaticamente

## Como Usar

### Após a Instalação

1. **Localize o atalho:**
   - Na área de trabalho: "CardioAI Pro"
   - No menu Iniciar: "CardioAI Pro"

2. **Inicie o sistema:**
   - Clique duas vezes no atalho
   - Aguarde alguns segundos para o sistema inicializar
   - A interface web abrirá automaticamente no seu navegador

3. **Acesso direto:**
   - Se a interface não abrir automaticamente
   - Abra seu navegador e acesse: `http://localhost:8000`

### Primeira Utilização

1. **Login inicial:**
   - Usuário: `admin`
   - Senha: (gerada automaticamente - veja logs do sistema)

2. **Altere a senha:**
   - Acesse as configurações do usuário
   - Defina uma nova senha segura

## Funcionalidades

### ✅ Incluído na Versão Standalone

- ✓ Análise automática de ECG com IA
- ✓ Interface web responsiva
- ✓ Processamento local (sem internet)
- ✓ Banco de dados SQLite integrado
- ✓ Modelos de IA pré-treinados
- ✓ Relatórios em PDF
- ✓ Compliance médico básico

### ❌ Não Incluído

- ❌ Sincronização em nuvem
- ❌ Backup automático remoto
- ❌ Atualizações automáticas
- ❌ Integração com sistemas hospitalares

## Solução de Problemas

### Problemas Comuns

**1. "Arquivo não encontrado" durante instalação**
- Certifique-se de que ambos os arquivos estão na mesma pasta
- Verifique se o download foi concluído corretamente

**2. "Acesso negado" ou "Privilégios insuficientes"**
- Execute como administrador
- Clique com botão direito > "Executar como administrador"

**3. Interface não abre automaticamente**
- Abra seu navegador manualmente
- Acesse: `http://localhost:8000`
- Verifique se o firewall não está bloqueando

**4. "Porta em uso" ou erro de conexão**
- Feche outros programas que possam usar a porta 8000
- Reinicie o computador e tente novamente

**5. Sistema lento ou travando**
- Verifique se há memória RAM suficiente
- Feche outros programas pesados
- Considere reiniciar o sistema

### Desinstalação

**Método 1: Painel de Controle**
1. Painel de Controle > Programas > Desinstalar um programa
2. Localize "CardioAI Pro"
3. Clique em "Desinstalar"

**Método 2: Menu Iniciar**
1. Menu Iniciar > CardioAI Pro > Uninstall
2. Confirme a desinstalação

## Suporte Técnico

### Informações do Sistema
- **Versão:** 1.0.0 Standalone
- **Tipo:** Aplicação desktop local
- **Banco de dados:** SQLite
- **Servidor:** FastAPI integrado

### Contato
- **GitHub:** https://github.com/drguilhermecapel/cardio.ai.pro
- **Issues:** https://github.com/drguilhermecapel/cardio.ai.pro/issues

### Logs e Diagnóstico

Para reportar problemas, inclua as seguintes informações:
- Versão do Windows
- Mensagem de erro completa
- Passos que levaram ao problema
- Screenshot da tela de erro (se aplicável)

## Segurança e Privacidade

- ✓ Todos os dados permanecem no seu computador
- ✓ Nenhuma informação é enviada para servidores externos
- ✓ Processamento 100% local
- ✓ Compliance com LGPD para dados locais

## Atualizações

Para atualizar o sistema:
1. Baixe a nova versão do instalador
2. Execute a nova instalação
3. O sistema manterá seus dados existentes

---

**Nota:** Esta é uma versão standalone simplificada. Para funcionalidades avançadas como integração hospitalar, backup em nuvem e suporte técnico dedicado, considere a versão empresarial.
