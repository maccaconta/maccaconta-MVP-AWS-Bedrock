import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def main():
    try:
        # cria sessão padrão (usa ~/.aws/credentials ou variáveis de ambiente)
        session = boto3.Session()

        # STS é global e sempre disponível
        sts = session.client("sts", region_name="us-east-1")

        identity = sts.get_caller_identity()

        print("✅ Conectado com sucesso à AWS")
        print("Account:", identity["Account"])
        print("User ARN:", identity["Arn"])

    except NoCredentialsError:
        print("❌ Nenhuma credencial encontrada")
    except ClientError as e:
        print("❌ Erro de permissão:", e.response["Error"]["Message"])
    except Exception as e:
        print("❌ Erro inesperado:", str(e))

if __name__ == "__main__":
    main()