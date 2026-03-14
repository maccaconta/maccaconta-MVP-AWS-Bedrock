# =====================================================
# EMS MVP - Makefile para Deploy em AWS EKS
# * Instalar: sudo apt install make
#
# * Criar Infraestrutura
# make infra-init
# make infra-plan ENV=dev
# make infra-apply ENV=dev
#
# * Fazer deploy da aplicação
# make deploy ENV=dev
#
# * Atualizar apenas a aplicação
# make update ENV=dev
#
# * Ver status
# make status ENV=dev
#
# * Ver Logs
# make logs ENV=dev
#
# * Rollback de release
# make rollback ENV=dev REVISION=2
#
# * Apagar infraestrutura
# make infra-destroy ENV=dev
# =====================================================

ENV ?= dev
TF_DIR=terraform
HELM_DIR=helm/ems-mvp
ENV_FILE=scripts/ems_mvp.env
SCRIPT_UPDATE=scripts/update_app_only.sh

TF_VARS=$(TF_DIR)/terraform-$(ENV).tfvars
VALUES_FILE=$(HELM_DIR)/values-$(ENV).yaml

APP_NAME=ems-mvp

# =====================================================
# Ajuda
# =====================================================

help:
	@echo ""
	@echo "Comandos disponíveis:"
	@echo ""
	@echo "Infraestrutura:"
	@echo "  make infra-init ENV=dev"
	@echo "  make infra-plan ENV=dev"
	@echo "  make infra-apply ENV=dev"
	@echo ""
	@echo "Deploy da aplicação:"
	@echo "  make deploy ENV=dev"
	@echo ""
	@echo "Atualizar apenas aplicação:"
	@echo "  make update ENV=dev"
	@echo ""
	@echo "Rollback:"
	@echo "  make rollback ENV=dev REVISION=<numero>"
	@echo ""
	@echo "Status:"
	@echo "  make status ENV=dev"
	@echo ""

# =====================================================
# Terraform - Infraestrutura
# =====================================================

infra-init:
	cd $(TF_DIR) && terraform init

infra-plan:
	cd $(TF_DIR) && terraform plan -var-file=$(TF_VARS)

infra-apply:
	cd $(TF_DIR) && terraform apply -var-file=$(TF_VARS)

infra-destroy:
	cd $(TF_DIR) && terraform destroy -var-file=$(TF_VARS)

# =====================================================
# Deploy via Helm
# =====================================================

deploy:
	helm upgrade --install $(APP_NAME) $(HELM_DIR) \
	-n $(ENV) --create-namespace \
	-f $(VALUES_FILE)

# =====================================================
# Atualização da aplicação (sem mexer na infra)
# =====================================================

update:
	$(SCRIPT_UPDATE) $(ENV_FILE)

# =====================================================
# Rollback
# =====================================================

rollback:
	helm rollback $(APP_NAME) $(REVISION) -n $(ENV)

# =====================================================
# Status
# =====================================================

status:
	kubectl get pods -n $(ENV)
	kubectl get svc -n $(ENV)

# =====================================================
# Logs
# =====================================================

logs:
	kubectl logs -n $(ENV) -l app=$(APP_NAME) --tail=100