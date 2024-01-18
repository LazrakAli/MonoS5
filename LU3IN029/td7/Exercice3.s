.data

n:	.word	15
m:	.word	-1
l:	.word	124

.text
# main
	addiu $29, $29, -24 # nv = 1 (variable locale tmp optimisee dans $16)
# na = 5 (4 premiers arguments = 4 mots + 1 mot pour 5eme arg)
# tmp = moyenne3(n, m, 5);
	lui $3, 0x1001
	lw $4, 0($3) # 1er param = valeur de n
	lw $5, 4($3) # 2eme param = valeur de m
	ori $6, $0, 5 # 3eme param = 5
	jal moyenne3
adr1:
	ori $16, $2, 0 # tmp = resultat
#printf("%d", tmp);
	or $4, $16, $0 # affichage de tmp
	ori $2, $0, 1
	syscall # tmp est potenetiellement ecrase
#tmp = moyenne5(m, l, m+5, 12, 35);
	lui $3, 0x1001 # $3 non persistant on doit recharger sa valeuer
	lw $4, 4($3) # 1er param = valeur de m	
	lw $5, 8($3) # 2eme param = valeur de l
	addiu $6, $4, 5 # 3eme param = m + 5
	ori $7, $0, 12 # 4eme param = 12
	ori $8, $0, 35 # 5eme param vaut 35
	sw $8, 16($29) # mise en pile 5eme param
	jal moyenne5
adr2:
	ori $16, $2, 0 # tmp = resultat
# printf("%d", tmp);
	or $4, $16, $0 # affichage du resultat
	ori $2, $0, 1
	syscall
# desallocation var sur la pile + exit();
	addiu $29, $29, +24
	ori $2, $0, 10
	syscall

# Q1: un parametre sera passé par la pile
moyenne3:
	addiu $29, $29, -12
	sw $31, 8($29)
	sw $16 4($29)
# @sum = $29 mais optimisee dans le registre $16
# sum = p + q + r;
	addu $8, $4, $5
	addu $16, $8, $6 # sum optimisee dans $16
	# return sum / 3;
	ori $9, $0, 3
	div $16, $9
# resultat dans $2
	mflo $2
	lw $16, 4($29)
	lw $31, 8($29)
	addiu $29, $29, 12
	jr $31	


moyenne5:
	addiu $29, $29, -12
	sw $31, 8($29)
	sw $16 4($29)
# @sum = $29 mais sum est optimisee dans le registre $16
# lecture du 5eme parametre s dans $10
	lw $10, 28($29)
# sum = p + q + r + s + t;
	addu $8, $4, $5 # P + q
	addu $8, $8, $6 # p + q + r
	addu $8, $8, $7 # p + q + r + s + t
	addu $16, $8, $10 # sum optimisee dans $16 = (p + q + r + s) + t
# return sum / 5;
	ori $9, $0, 5
	div $16, $9
# resultat dans $2
	mflo $2
	lw $16, 4($29)
	lw $31, 8($29)
	addiu $29, $29, 12
	jr $31