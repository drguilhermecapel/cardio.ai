# 🏥 CardioAI Pro v1.0.0 - Guia de Instalação Completo

## 🚀 Instalação Automática (Recomendado)

### Método 1: Instalador Interativo
```bash
# 1. Extrair o arquivo ZIP
unzip cardioai-pro-v1.0.0.zip
cd cardioai-pro-v1.0.0/

# 2. Executar o instalador
./docker-compose up -d --build
```

O instalador irá:
- ✅ Verificar pré-requisitos automaticamente
- ⚙️ Configurar ambiente com interface amigável
- 🐳 Instalar e configurar Docker containers
- 🔐 Configurar segurança e compliance médico
- 🌐 Inicializar sistema completo
- ✨ Criar usuário administrador

### Método 2: Instalação Manual Rápida
```bash
# 1. Extrair e entrar no diretório
unzip cardioai-pro-v1.0.0.zip
cd cardioai-pro-v1.0.0/

# 2. Configurar ambiente
cp .env.example .env
nano .env  # Editar configurações

# 3. Iniciar sistema
docker-compose up -d --build

# 4. Aguardar inicialização (2-3 minutos)
docker-compose logs -f
```

## 📋 Pré-requisitos

### Sistema Operacional
- ✅ Ubuntu 20.04+ / Debian 11+
- ✅ CentOS 8+ / RHEL 8+
- ✅ Windows 10+ (com WSL2)
- ✅ macOS 11+

## 🪟 Instalação no Windows

### Método 1: Instalação Automática com PowerShell (Recomendado)

**Pré-requisitos:**
- Windows 10 ou superior
- PowerShell 5.1+ (incluído no Windows 10/11)
- Privilégios de administrador
- Conexão com internet

**Passos:**
1. **Baixe os arquivos do repositório**
   ```
   https://github.com/drguilhermecapel/cardio.ai.pro
   ```

2. **Execute o instalador automático**
   - Baixe `CardioAI-Pro-1.0.0-installer.exe`
   - Clique com o botão direito e escolha "Executar como administrador"
   - Siga as instruções na tela

### Método 1B: Instalação Manual (Sem PowerShell)

**Para sistemas sem PowerShell ou com problemas de execução:**

**Passos:**
1. **Baixe os arquivos do repositório**
   ```
   https://github.com/drguilhermecapel/cardio.ai.pro
   ```

2. **Compile manualmente a partir do código fonte**
   - Execute os scripts em `windows_installer/`:
     ```cmd
     python build_backend.py
     python build_frontend.py
     makensis cardioai_installer.nsi
     ```
   - Use o instalador gerado em `frontend/dist-electron/`

3. **Aguarde a instalação automática**
   - O instalador irá automaticamente:
     - Verificar e instalar WSL2
     - Baixar e instalar Docker Desktop
     - Configurar Ubuntu no WSL2
     - Instalar o CardioAI Pro

### Método 2: Instalação via PowerShell

**Execute como Administrador:**
```powershell
# Navegue até o diretório do projeto
cd C:\caminho\para\cardio.ai.pro

# Execute o instalador
CardioAI-Pro-1.0.0-installer.exe
```

### O que o Instalador Windows Faz Automaticamente

1. **Verificação de Sistema**
   - Confirma Windows 10+ 
   - Verifica privilégios de administrador
   - Testa conectividade de internet

2. **Instalação WSL2**
   - Habilita recurso Windows Subsystem for Linux
   - Habilita plataforma de máquina virtual
   - Baixa e instala atualização do kernel WSL2
   - Define WSL2 como versão padrão

3. **Instalação Docker Desktop**
   - Baixa automaticamente a versão mais recente
   - Instala silenciosamente
   - Configura integração com WSL2

4. **Configuração Ubuntu**
   - Instala distribuição Ubuntu no WSL2
   - Configura ambiente Linux

5. **Instalação CardioAI Pro**
   - Clona o repositório no ambiente WSL2
   - Executa o instalador Linux dentro do WSL2
   - Configura todos os serviços

### Informações Pós-Instalação (Windows)

**URLs de Acesso:**
- Frontend: http://localhost:3000
- API: http://localhost:8000  
- Documentação: http://localhost:8000/docs

**Credenciais Padrão:**
- Usuário: admin@cardioai.pro
- Senha: (gerada automaticamente no primeiro uso - veja logs do sistema)

**Comandos Úteis (PowerShell):**
```powershell
# Ver logs do sistema
wsl -d Ubuntu -e docker-compose logs -f

# Parar o sistema
wsl -d Ubuntu -e docker-compose down

# Reiniciar o sistema  
wsl -d Ubuntu -e docker-compose restart

# Acessar terminal Ubuntu
wsl -d Ubuntu

# Ver status dos containers
wsl -d Ubuntu -e docker-compose ps
```

**Localização dos Arquivos:**
- Arquivos do projeto: `\\wsl$\Ubuntu\tmp\cardio.ai.pro`
- Logs: `\\wsl$\Ubuntu\tmp\cardio.ai.pro\logs`

### Solução de Problemas (Windows)

#### Problema: "WSL2 não está disponível"
**Solução:**
1. Verifique se a virtualização está habilitada no BIOS
2. Execute como administrador:
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```
3. Reinicie o computador
4. Execute o instalador novamente

#### Problema: "Docker Desktop não inicia"
**Solução:**
1. Verifique se o Hyper-V está habilitado
2. Reinicie o Docker Desktop manualmente
3. Verifique se há conflitos com outros softwares de virtualização

#### Problema: "Erro de permissão"
**Solução:**
1. Certifique-se de executar como administrador
2. Desabilite temporariamente o antivírus
3. Verifique as políticas de execução do PowerShell:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

#### Problema: "Ubuntu não responde"
**Solução:**
1. Reinicie o WSL2:
   ```powershell
   wsl --shutdown
   wsl -d Ubuntu
   ```
2. Verifique se o Docker está rodando:
   ```powershell
   docker ps
   ```

#### Problema: "Portas já em uso"
**Solução:**
1. Verifique processos usando as portas:
   ```powershell
   netstat -ano | findstr :3000
   netstat -ano | findstr :8000
   ```
2. Termine processos conflitantes ou altere as portas no docker-compose.yml

### Reinstalação (Windows)

Para reinstalar completamente:

1. **Parar todos os serviços:**
   ```powershell
   wsl -d Ubuntu -e docker-compose down
   ```

2. **Remover containers e volumes:**
   ```powershell
   wsl -d Ubuntu -e docker system prune -a --volumes
   ```

3. **Executar instalador novamente:**
   ```powershell
   CardioAI-Pro-1.0.0-installer.exe
   ```

### Software Necessário
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose curl

# CentOS/RHEL
sudo yum install docker docker-compose curl
sudo systemctl start docker
sudo systemctl enable docker

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
# Fazer logout/login após este comando
```

### Recursos Mínimos
- 🖥️ **CPU**: 2 cores (4 cores recomendado)
- 🧠 **RAM**: 4GB (8GB recomendado)
- 💾 **Disco**: 10GB livres
- 🌐 **Rede**: Portas 3000 e 8000 disponíveis

## ⚙️ Configuração Detalhada

### Arquivo .env Principal
```bash
# Copiar template
cp .env.example .env

# Configurações essenciais
ENVIRONMENT=production
DEBUG=false

# Banco de Dados
DATABASE_URL=postgresql://postgres:SUA_SENHA_AQUI@postgres:5432/cardioai_pro
POSTGRES_PASSWORD=SUA_SENHA_AQUI

# Segurança (GERAR CHAVES ÚNICAS!)
SECRET_KEY=sua-chave-secreta-super-segura-aqui
JWT_SECRET_KEY=sua-chave-jwt-super-segura-aqui

# Portas (alterar se necessário)
API_PORT=8000
WEB_PORT=3000

# Email (opcional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=seu-email@gmail.com
SMTP_PASSWORD=sua-senha-app
```

### Configurações de Produção
```bash
# Para ambiente de produção, adicione:
ALLOWED_ORIGINS=https://seu-dominio.com
SSL_ENABLED=true
SECURE_COOKIES=true

# Compliance médico
MEDICAL_COMPLIANCE_MODE=true
AUDIT_LOGGING=true
DATA_RETENTION_DAYS=2555  # 7 anos ANVISA
```

## 🐳 Comandos Docker

### Inicialização
```bash
# Primeira execução (com build)
docker-compose up -d --build

# Execuções subsequentes
docker-compose up -d

# Ver logs em tempo real
docker-compose logs -f

# Ver logs de serviço específico
docker-compose logs -f api
docker-compose logs -f frontend
```

### Manutenção
```bash
# Parar sistema
docker-compose down

# Parar e remover volumes (CUIDADO!)
docker-compose down -v

# Reiniciar serviços
docker-compose restart

# Atualizar imagens
docker-compose pull
docker-compose up -d --build

# Ver status
docker-compose ps
```

### Backup e Restore
```bash
# Backup do banco
docker-compose exec postgres pg_dump -U postgres cardioai_pro > backup_$(date +%Y%m%d).sql

# Restore do banco
docker-compose exec -T postgres psql -U postgres cardioai_pro < backup_20250602.sql

# Backup completo (volumes)
docker run --rm -v cardioai_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/volumes_backup.tar.gz /data
```

## 🔍 Verificação da Instalação

### Testes de Conectividade
```bash
# API Health Check
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# Documentação da API
curl http://localhost:8000/docs
```

### Verificação de Serviços
```bash
# Status dos containers
docker-compose ps

# Logs de erro
docker-compose logs | grep -i error

# Uso de recursos
docker stats

# Verificar banco
docker-compose exec postgres pg_isready -U postgres
```

## 🌐 Acesso ao Sistema

### URLs Principais
- 🏠 **Frontend**: http://localhost:3000
- 🔌 **API**: http://localhost:8000
- 📚 **Documentação**: http://localhost:8000/docs
- 🔄 **API Redoc**: http://localhost:8000/redoc

### Credenciais Padrão
```
Usuário: admin@cardioai.pro
Senha: (gerada automaticamente no primeiro uso)
```
⚠️ **IMPORTANTE**: Verifique os logs do sistema para obter a senha inicial e altere-a após o primeiro login!

### Criação de Usuários
```bash
# Via interface web (recomendado)
# Acesse: http://localhost:3000/admin

# Via linha de comando
docker-compose exec api python -c "
from app.scripts.create_user import create_user
import asyncio
asyncio.run(create_user('medico@hospital.com', 'senha123', 'physician'))
"
```

## 🔧 Solução de Problemas

### Problemas Comuns

#### Portas Ocupadas
```bash
# Verificar portas em uso
sudo netstat -tulpn | grep :3000
sudo netstat -tulpn | grep :8000

# Alterar portas no .env
API_PORT=8001
WEB_PORT=3001
```

#### Erro de Permissão Docker
```bash
# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER

# Fazer logout/login
exit
# Fazer login novamente
```

#### Containers Não Iniciam
```bash
# Ver logs detalhados
docker-compose logs

# Limpar cache Docker
docker system prune -a

# Reconstruir do zero
docker-compose down -v
docker-compose up -d --build
```

#### Banco de Dados Não Conecta
```bash
# Verificar se PostgreSQL está rodando
docker-compose ps postgres

# Verificar logs do banco
docker-compose logs postgres

# Resetar banco (CUIDADO!)
docker-compose down
docker volume rm cardioai_postgres_data
docker-compose up -d
```

#### Frontend Não Carrega
```bash
# Verificar build do frontend
docker-compose logs frontend

# Reconstruir frontend
docker-compose build frontend
docker-compose up -d frontend

# Verificar nginx
docker-compose exec frontend nginx -t
```

### Logs e Debugging
```bash
# Logs completos
docker-compose logs --tail=100

# Logs em tempo real
docker-compose logs -f

# Logs de erro apenas
docker-compose logs | grep -i "error\|exception\|failed"

# Entrar no container para debug
docker-compose exec api bash
docker-compose exec frontend sh
```

### Performance
```bash
# Monitorar recursos
docker stats

# Verificar espaço em disco
df -h
docker system df

# Limpar logs antigos
docker-compose logs --tail=0 -f > /dev/null &
```

## 🔐 Segurança e Compliance

### Configurações de Segurança
```bash
# No arquivo .env
SECURE_COOKIES=true
SESSION_TIMEOUT=1800
MAX_LOGIN_ATTEMPTS=5
PASSWORD_MIN_LENGTH=8

# HTTPS (produção)
SSL_ENABLED=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### Compliance Médico
- ✅ **ANVISA RDC 185/2001**: Retenção de dados por 7 anos
- ✅ **FDA 21 CFR Part 11**: Assinatura eletrônica
- ✅ **LGPD**: Proteção de dados pessoais
- ✅ **HIPAA**: Segurança de informações de saúde
- ✅ **IEC 62304**: Software médico Classe C

### Auditoria
```bash
# Logs de auditoria
docker-compose exec api tail -f /app/logs/audit.log

# Relatório de compliance
docker-compose exec api python -m app.scripts.compliance_report
```

## 📊 Monitoramento

### Métricas do Sistema
```bash
# Prometheus (se habilitado)
curl http://localhost:9090/metrics

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
```

### Alertas
```bash
# Configurar alertas por email
ALERT_EMAIL=admin@hospital.com
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=85
```

## 🔄 Atualizações

### Atualização do Sistema
```bash
# Backup antes da atualização
./backup.sh

# Baixar nova versão
wget https://github.com/drguilhermecapel/cardio.ai.pro/releases/latest/download/cardioai-pro-latest.zip

# Atualizar
unzip cardioai-pro-latest.zip
docker-compose down
docker-compose pull
docker-compose up -d --build
```

### Migração de Dados
```bash
# Executar migrações
docker-compose exec api alembic upgrade head

# Verificar versão do banco
docker-compose exec api alembic current
```

## 📞 Suporte

### Documentação
- 📖 **Manual**: http://localhost:8000/docs
- 🔗 **GitHub**: https://github.com/drguilhermecapel/cardio.ai.pro
- 🐛 **Issues**: https://github.com/drguilhermecapel/cardio.ai.pro/issues

### Contato
- 📧 **Email**: suporte@cardioai.pro
- 💬 **Chat**: Disponível na interface web
- 📱 **WhatsApp**: +55 11 99999-9999

---

## ✅ Checklist de Instalação

- [ ] Pré-requisitos verificados
- [ ] Docker e Docker Compose instalados
- [ ] Arquivo .env configurado
- [ ] Sistema iniciado com docker-compose
- [ ] Health checks passando
- [ ] Frontend acessível
- [ ] API respondendo
- [ ] Usuário admin criado
- [ ] Backup configurado
- [ ] Monitoramento ativo

**🎉 Parabéns! Seu CardioAI Pro está pronto para uso! 🏥⚡**
