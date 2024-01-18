.data

ch:	.space	11
.text
	addiu	$29,$29,-16

# scanf("%d", &nb); 
	ori	$2,$0,5
	syscall
	ori	$10,$2,0
	sw	$10,4($29)
	
# chaine[10] = 0;
	lui	$8,0x1001
	addiu	$8,$8,10
	sb	$0,0($8)
# i = 9;
	ori	$9,$0,9
	sw	$9,0($29)
# r15<-10
	ori	$15,$0,10
for:
# decrementation de l'adresse
	addiu	$8,$8,-1
# r = nb % 10;
	div	$10,$15
	mfhi	$11
	sw	$11,8($29)
# r = nb % 10;
	mflo	$10
	sw	$10,4($29)
# chaine[i] = r + 0x30;
	xor	$13,$13,$13
	addiu	$13,$11,0x30
	sb	$13,0($8)


	sw	$9,0($29)
	blez	$9,endfor
#i -= 1
	addiu	$9,$9,-1
	j for
	
endfor:
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
	
	addiu	$29,$29,16
	
	ori	$2,$0,10
	syscall
	
	
	